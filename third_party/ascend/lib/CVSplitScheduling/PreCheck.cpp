/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/CVSplitScheduling/PreCheck.h"
#include "ascend/include/CVSplitScheduling/HardwareConstants.h"
#include "ascend/include/CVSplitScheduling/UnfusePVMatmuls.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "cv-split-pre-check"

using namespace mlir;
using mlir::triton::cv_split::kNzTileSize;
namespace {

constexpr llvm::StringLiteral kLoopCarriedConsumersAttr =
    "cv_split.loop_carried_consumers";

static SmallVector<scf::ForOp> collectInnermostLoops(func::FuncOp funcOp) {
  SmallVector<scf::ForOp> loops;
  funcOp.walk([&](scf::ForOp forOp) {
    bool hasNestedLoop = false;
    forOp.getBody()->walk([&](scf::ForOp) {
      hasNestedLoop = true;
      return WalkResult::interrupt();
    });
    if (!hasNestedLoop)
      loops.push_back(forOp);
  });
  return loops;
}

static bool containsStore(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    if (isa<memref::StoreOp, tensor::InsertSliceOp,
            bufferization::MaterializeInDestinationOp>(op))
      return WalkResult::interrupt();

    if (op->getName().getStringRef().contains("store"))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static bool containsBranching(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    if (isa<RegionBranchOpInterface>(op) || op->getNumSuccessors() != 0)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static LogicalResult checkStaticTensorType(Type type) {
  if (!isa<TensorType>(type))
    return success();

  auto rankedType = dyn_cast<RankedTensorType>(type);
  return success(rankedType && rankedType.hasStaticShape());
}

static LogicalResult checkStaticTensorShapes(scf::ForOp forOp) {
  auto checkValue = [](Value value) {
    return checkStaticTensorType(value.getType());
  };

  for (Value initArg : forOp.getInitArgs())
    if (failed(checkValue(initArg)))
      return failure();

  for (Value result : forOp.getResults())
    if (failed(checkValue(result)))
      return failure();

  for (BlockArgument argument : forOp.getBody()->getArguments())
    if (failed(checkValue(argument)))
      return failure();

  WalkResult walkResult = forOp.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (failed(checkValue(operand)))
        return WalkResult::interrupt();

    for (Value result : op->getResults())
      if (failed(checkValue(result)))
        return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

static LogicalResult checkMatmulOutsAreRankedTensors(scf::ForOp forOp) {
  for (Operation &op : *forOp.getBody()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp)
      continue;

    Value outs = matmulOp.getDpsInitOperand(0)->get();
    if (!isa<RankedTensorType>(outs.getType()))
      return failure();
  }
  return success();
}

static bool containsMatmul(scf::ForOp forOp) {
  return llvm::any_of(*forOp.getBody(),
                      [](Operation &op) { return isa<linalg::MatmulOp>(op); });
}

static LogicalResult checkMatmulDimensionsAre16Aligned(scf::ForOp forOp) {
  auto is16AlignedRank2Tensor = [](Type type) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || tensorType.getRank() != 2)
      return false;
    for (int64_t dim : tensorType.getShape())
      if (dim <= 0 || dim % kNzTileSize != 0)
        return false;
    return true;
  };

  for (Operation &op : *forOp.getBody()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp)
      continue;

    for (Value operand : matmulOp->getOperands())
      if (!is16AlignedRank2Tensor(operand.getType()))
        return failure();
    for (Value result : matmulOp->getResults())
      if (!is16AlignedRank2Tensor(result.getType()))
        return failure();
  }
  return success();
}

// ROW_SPLIT assigns M/2 rows to each vector core. The subsequent V->C NZ pack
// groups those per-core rows in blocks of 16, so the original M must be a
// multiple of 2 * 16 = 32.
static LogicalResult checkMatmulRowsAre32Aligned(scf::ForOp forOp) {
  for (Operation &op : *forOp.getBody()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp)
      continue;

    auto resultType =
        dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
    if (!resultType || resultType.getRank() != 2 ||
        resultType.getShape()[0] <= 0 ||
        resultType.getShape()[0] % (2 * kNzTileSize) != 0)
      return failure();
  }
  return success();
}

static LogicalResult checkStaticDivisibleTripCount(scf::ForOp forOp,
                                                   int unrollFactor) {
  std::optional<int64_t> lowerBound =
      getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> upperBound =
      getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lowerBound || !upperBound || !step || *step <= 0 ||
      *upperBound <= *lowerBound)
    return failure();

  int64_t distance;
  if (llvm::SubOverflow(*upperBound, *lowerBound, distance))
    return failure();

  int64_t tripCount = distance / *step + (distance % *step != 0);
  return success(tripCount >= unrollFactor && tripCount % unrollFactor == 0);
}

} // namespace

FailureOr<scf::ForOp>
mlir::triton::cv_split::preCheckCVSplitScheduling(func::FuncOp funcOp,
                                                  int unrollFactor) {
  if (unrollFactor != 2 && unrollFactor != 4 && unrollFactor != 8) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] unsupported unroll factor: "
               << unrollFactor << "\n");
    return failure();
  }

  SmallVector<scf::ForOp> candidates = collectInnermostLoops(funcOp);
  if (candidates.size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] expected one innermost loop in "
               << funcOp.getName() << ", found " << candidates.size() << "\n");
    return failure();
  }

  // this is the chosen for loop
  scf::ForOp candidate = candidates.front();

  if (failed(checkStaticTensorShapes(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains an unranked "
                  "or dynamically shaped tensor\n");
    return failure();
  }

  if (!containsMatmul(candidate)) {
    LLVM_DEBUG(llvm::dbgs() << "[cv-split-pre-check] candidate loop must "
                               "contain at least one linalg.matmul\n");
    return failure();
  }

  if (failed(checkMatmulOutsAreRankedTensors(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] linalg.matmul outs must be a ranked "
                  "tensor\n");
    return failure();
  }

  if (failed(checkMatmulDimensionsAre16Aligned(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] linalg.matmul operands and results "
                  "must be rank-2 tensors with dimensions divisible by 16\n");
    return failure();
  }

  if (failed(checkMatmulRowsAre32Aligned(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] linalg.matmul output rows must be "
                  "divisible by 32 for two-core ROW_SPLIT\n");
    return failure();
  }

  if (failed(checkStaticDivisibleTripCount(candidate, unrollFactor))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop must have a static, "
                  "positive trip count divisible by the unroll factor\n");
    return failure();
  }

  if (containsBranching(candidate)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains branching\n");
    return failure();
  }

  if (containsStore(candidate)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains a store\n");
    return failure();
  }

  return candidate;
}

void mlir::triton::cv_split::markLoopCarriedConsumers(scf::ForOp loop) {
  // Store operand numbers, not SSA names or operation order. Unrolling remaps
  // the values but preserves these operand positions on cloned operations.
  loop.walk([&](Operation *operation) {
    if (operation == loop.getOperation() || isa<scf::YieldOp>(operation))
      return;
    SmallVector<int64_t> operands;
    for (OpOperand &use : operation->getOpOperands()) {
      auto argument = dyn_cast<BlockArgument>(use.get());
      if (argument && argument.getOwner() == loop.getBody() &&
          argument.getArgNumber() != 0)
        operands.push_back(use.getOperandNumber());
    }
    if (!operands.empty())
      operation->setAttr(kLoopCarriedConsumersAttr,
                         DenseI64ArrayAttr::get(loop.getContext(), operands));
  });
}

void mlir::triton::cv_split::removeLoopCarriedConsumerMarkers(Operation *root) {
  root->walk([](Operation *operation) {
    operation->removeAttr(kLoopCarriedConsumersAttr);
  });
}

LogicalResult mlir::triton::cv_split::preCheckCrossIterationConsumers(
    scf::ForOp loop, const Classification &classification,
    const AccumulatorJoinRewriteResult &accumulatorJoins) {
  if (!loop)
    return failure();
  Block *body = loop.getBody();
  auto yield = dyn_cast<scf::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getNumRegionIterArgs())
    return failure();

  // Full unrolling keeps one scaffold iteration. Its initial arguments do not
  // consume the final yield; the real inter-iteration edges are now marked SSA
  // edges in the body. Avoid inventing an extra execution of that scaffold.
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  int64_t distance = 0;
  bool singleIteration = lower && upper && step && *step > 0 &&
                         !llvm::SubOverflow(*upper, *lower, distance) &&
                         distance > 0 && distance <= *step;

  auto reject = [&](Operation *consumer, StringRef reason) -> LogicalResult {
    consumer->emitRemark() << "CVSplit cross-iteration pre-check rejected: "
                           << reason;
    return failure();
  };
  // Scope separation deliberately reproduces scalar/index control in both
  // scopes, irrespective of its provisional single-core stamp. Accept only
  // a closed, pure scalar recurrence: a scalar extracted from a tensor or
  // loaded from memory is not made safe merely by having a scalar result.
  auto isReplicatedScalar = [&](Value value) {
    SmallVector<Value> worklist{value};
    llvm::DenseSet<Value> seen;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!current.getType().isIntOrIndexOrFloat())
        return false;
      if (!seen.insert(current).second)
        continue;
      if (auto argument = dyn_cast<BlockArgument>(current)) {
        if (argument.getOwner() == body && argument.getArgNumber() != 0) {
          unsigned index = argument.getArgNumber() - 1;
          worklist.push_back(loop.getInitArgs()[index]);
          if (!singleIteration)
            worklist.push_back(yield.getOperand(index));
        }
        continue;
      }
      Operation *definition = current.getDefiningOp();
      if (!definition || !loop->isAncestor(definition))
        continue;
      if (definition->getNumRegions() != 0 || !isMemoryEffectFree(definition) ||
          !llvm::all_of(definition->getResultTypes(),
                        [](Type type) { return type.isIntOrIndexOrFloat(); }))
        return false;
      worklist.append(definition->operand_begin(), definition->operand_end());
    }
    return true;
  };
  auto checkUse = [&](OpOperand &use) -> LogicalResult {
    Operation *consumer = body->findAncestorOpInBlock(*use.getOwner());
    if (!consumer || consumer == yield.getOperation())
      return success();
    auto consumerCore = classification.find(consumer);
    if (consumerCore == classification.end())
      return reject(consumer, "consumer has no single-core classification");
    if (isReplicatedScalar(use.get()))
      return success();

    Value incoming = use.get();
    llvm::DenseSet<Value> visited;
    while (auto argument = dyn_cast<BlockArgument>(incoming)) {
      if (argument.getOwner() != body || argument.getArgNumber() == 0)
        return success(); // Outer value or loop induction value.
      if (!visited.insert(incoming).second)
        return success(); // A pure permutation of invariant initial values.
      unsigned index = argument.getArgNumber() - 1;
      incoming =
          singleIteration ? loop.getInitArgs()[index] : yield.getOperand(index);
    }
    Operation *producer = incoming.getDefiningOp();
    if (!producer || !loop->isAncestor(producer))
      return success(); // No value produced in this loop crosses an iteration.
    producer = body->findAncestorOpInBlock(*producer);
    if (!producer)
      return reject(consumer, "cannot bind the carried value's producer");
    auto producerCore = classification.find(producer);
    if (producerCore == classification.end())
      return reject(consumer, "producer has no single-core classification");
    if (producerCore->second == consumerCore->second)
      return success();
    consumer->emitRemark() << "CVSplit cross-iteration pre-check rejected: "
                           << (producerCore->second == EngineType::CUBE
                                   ? "CUBE -> VECTOR"
                                   : "VECTOR -> CUBE")
                           << " carried value from " << producer->getName()
                           << " to " << use.getOwner()->getName() << " operand "
                           << use.getOperandNumber();
    return failure();
  };

  // An unfused DPS accumulator is now consumed by the VECTOR add, not by the
  // CUBE matmul's new zero destination. Reuse the actual rewrite binding.
  llvm::DenseMap<Operation *, Operation *> joins;
  for (const AccumulatorJoinBinding &binding : accumulatorJoins.bindings)
    joins.try_emplace(binding.matmulProducer, binding.vectorJoin);

  WalkResult marked = loop.walk([&](Operation *operation) -> WalkResult {
    auto operands =
        operation->getAttrOfType<DenseI64ArrayAttr>(kLoopCarriedConsumersAttr);
    if (!operands)
      return WalkResult::advance();
    for (int64_t index : operands.asArrayRef()) {
      if (index < 0 ||
          static_cast<uint64_t>(index) >= operation->getNumOperands()) {
        (void)reject(operation,
                     "invalid carried-operand binding after preparation");
        return WalkResult::interrupt();
      }
      OpOperand *use = &operation->getOpOperand(static_cast<unsigned>(index));
      if (Operation *join = joins.lookup(operation)) {
        auto dps = dyn_cast<DestinationStyleOpInterface>(operation);
        if (dps && dps.getNumDpsInits() == 1 &&
            use == dps.getDpsInitOperand(0)) {
          if (!isa<arith::AddFOp>(join) || join->getNumOperands() != 2) {
            (void)reject(operation, "invalid accumulator-join binding");
            return WalkResult::interrupt();
          }
          use = &join->getOpOperand(1);
        }
      }
      if (failed(checkUse(*use)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (marked.wasInterrupted())
    return failure();

  // Include state introduced during preparation (for example carried physical
  // addresses), and follow forwarded argument/yield chains with cycle safety.
  if (!singleIteration)
    for (BlockArgument argument : loop.getRegionIterArgs())
      for (OpOperand &use : argument.getUses())
        if (failed(checkUse(use)))
          return failure();
  return success();
}
