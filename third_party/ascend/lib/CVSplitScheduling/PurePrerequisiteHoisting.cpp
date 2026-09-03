/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "ascend/include/CVSplitScheduling/PurePrerequisiteHoisting.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static void collectSameBlockPredecessors(Operation *op, Block *body,
                                         DenseSet<Operation *> &slice) {
  if (!op || op->getBlock() != body || !slice.insert(op).second)
    return;
  for (Value operand : op->getOperands())
    collectSameBlockPredecessors(operand.getDefiningOp(), body, slice);
}

static DenseSet<Operation *> collectSameBlockDescendants(Value root,
                                                         Block *body) {
  DenseSet<Operation *> descendants;
  SmallVector<Operation *> worklist;
  for (Operation *user : root.getUsers())
    if (user->getBlock() == body && descendants.insert(user).second)
      worklist.push_back(user);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers())
        if (user->getBlock() == body && descendants.insert(user).second)
          worklist.push_back(user);
  }
  return descendants;
}

static FailureOr<uint64_t> getStaticScalarOrRankOneBytes(Type type) {
  Type elementType = type;
  uint64_t elements = 1;
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (!tensorType.hasStaticShape() || tensorType.getRank() > 1)
      return failure();
    const int64_t signedElements = tensorType.getNumElements();
    if (signedElements < 0)
      return failure();
    elements = static_cast<uint64_t>(signedElements);
    elementType = tensorType.getElementType();
  } else if (!isa<IntegerType, FloatType>(type)) {
    return failure();
  }

  unsigned bitWidth = 0;
  if (auto integerType = dyn_cast<IntegerType>(elementType))
    bitWidth = integerType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  if (bitWidth == 0)
    return failure();

  const uint64_t bytesPerElement = (static_cast<uint64_t>(bitWidth) + 7) / 8;
  if (bytesPerElement != 0 &&
      elements > std::numeric_limits<uint64_t>::max() / bytesPerElement)
    return failure();
  return elements * bytesPerElement;
}

static bool
isMovablePrerequisite(Operation *op, const Classification &classification,
                      const DenseSet<Operation *> &consumerSlice,
                      const DenseSet<Operation *> &transferredDescendants) {
  auto classIt = classification.find(op);
  if (classIt == classification.end() ||
      classIt->second != EngineType::VECTOR || !consumerSlice.contains(op) ||
      transferredDescendants.contains(op) || op->getNumRegions() != 0 ||
      op->getNumResults() == 0 || !isMemoryEffectFree(op))
    return false;

  return llvm::all_of(op->getResultTypes(), [](Type type) {
    return succeeded(getStaticScalarOrRankOneBytes(type));
  });
}

static void closeMovableOperands(Operation *wait, Block *body,
                                 DenseSet<Operation *> &selected) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> rejected;
    for (Operation *op : selected) {
      bool operandsAvailable = true;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def || def->getBlock() != body || def->isBeforeInBlock(wait))
          continue;
        if (!selected.contains(def)) {
          operandsAvailable = false;
          break;
        }
      }
      if (!operandsAvailable)
        rejected.push_back(op);
    }
    for (Operation *op : rejected)
      changed |= selected.erase(op);
  }
}

static FailureOr<uint64_t>
getAdditionalLiveBytes(const DenseSet<Operation *> &selected) {
  uint64_t total = 0;
  for (Operation *op : selected) {
    for (Value result : op->getResults()) {
      const bool escapes =
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !selected.contains(user);
          });
      if (!escapes)
        continue;
      FailureOr<uint64_t> bytes =
          getStaticScalarOrRankOneBytes(result.getType());
      if (failed(bytes) ||
          *bytes > std::numeric_limits<uint64_t>::max() - total)
        return failure();
      total += *bytes;
    }
  }
  return total;
}

static Operation *findFirstConsumer(const CubeToVectorTransferChain &chain,
                                    Block *body) {
  Operation *first = nullptr;
  for (Operation *consumer : chain.consumers) {
    if (!consumer || consumer->getBlock() != body)
      continue;
    if (!first || consumer->isBeforeInBlock(first))
      first = consumer;
  }
  return first;
}

} // namespace

LogicalResult hoistPurePrerequisites(Block *body,
                                     const Classification &classification,
                                     const CrossScopeTransferInfo &transferInfo,
                                     uint64_t budgetBytes) {
  if (!body || budgetBytes == 0)
    return failure();

  SmallVector<const CubeToVectorTransferChain *> chains;
  for (const CubeToVectorTransferChain &chain : transferInfo.cubeToVectorChains)
    chains.push_back(&chain);
  llvm::stable_sort(chains, [](const CubeToVectorTransferChain *lhs,
                               const CubeToVectorTransferChain *rhs) {
    return lhs->wait->isBeforeInBlock(rhs->wait);
  });

  uint64_t cumulativeLiveBytes = 0;
  unsigned movedOperations = 0;
  unsigned analyzedWaits = 0;
  for (const CubeToVectorTransferChain *chain : chains) {
    Operation *wait = chain->wait;
    Operation *firstConsumer = findFirstConsumer(*chain, body);
    Operation *transferredDef = chain->transferredValue
                                    ? chain->transferredValue.getDefiningOp()
                                    : nullptr;
    if (!wait || wait->getBlock() != body || !firstConsumer ||
        !wait->isBeforeInBlock(firstConsumer) || !chain->transferredValue ||
        !transferredDef || transferredDef->getBlock() != body)
      return failure();
    ++analyzedWaits;

    DenseSet<Operation *> consumerSlice;
    collectSameBlockPredecessors(firstConsumer, body, consumerSlice);
    DenseSet<Operation *> transferredDescendants =
        collectSameBlockDescendants(chain->transferredValue, body);

    DenseSet<Operation *> selected;
    bool afterWait = false;
    for (Operation &op : *body) {
      if (&op == wait) {
        afterWait = true;
        continue;
      }
      if (&op == firstConsumer)
        break;
      if (afterWait && isMovablePrerequisite(&op, classification, consumerSlice,
                                             transferredDescendants))
        selected.insert(&op);
    }
    closeMovableOperands(wait, body, selected);

    SmallVector<Operation *> ordered;
    for (Operation &op : *body)
      if (selected.contains(&op))
        ordered.push_back(&op);

    FailureOr<uint64_t> liveBytes = getAdditionalLiveBytes(selected);
    if (failed(liveBytes))
      return failure();
    const bool fits = *liveBytes <= budgetBytes - cumulativeLiveBytes;
    if (!fits) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cv-split] prerequisite-hoist origin=" << chain->originId
                 << " flag=" << chain->forwardFlagId << " considered="
                 << ordered.size() << " moved=0 live-bytes=" << *liveBytes
                 << " status=budget-rejected\n");
      continue;
    }

    for (Operation *op : ordered)
      op->moveBefore(wait);
    cumulativeLiveBytes += *liveBytes;
    movedOperations += ordered.size();
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split] prerequisite-hoist origin=" << chain->originId
               << " flag=" << chain->forwardFlagId
               << " considered=" << ordered.size()
               << " moved=" << ordered.size() << " live-bytes=" << *liveBytes
               << " status=" << (ordered.empty() ? "no-candidate" : "applied")
               << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "[cv-split] prerequisite-hoist-summary waits="
                          << analyzedWaits << " moved=" << movedOperations
                          << " live-bytes=" << cumulativeLiveBytes
                          << " budget=" << budgetBytes << "\n");
  return success();
}

} // namespace mlir::triton::cv_split
