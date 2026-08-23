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

#include "ascend/include/CVSplitScheduling/SoftmaxRegroup.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cv-split-scheduling"

using namespace mlir;

namespace mlir::triton::cv_split {

namespace {

/// One unrolled lane's share of the streaming-softmax recurrence.
struct Lane {
  linalg::ReduceOp rowMax;   ///< rowmax of this lane's score tile
  arith::MaximumFOp running; ///< max(previous running max, rowMax)
};

/// True when `op` reduces with a single `arith.maximumf`, i.e. it is a row max
/// rather than the row sum that shares its shape.
bool isMaxReduce(linalg::ReduceOp reduce) {
  Block &body = reduce.getCombiner().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  return isa_and_nonnull<arith::MaximumFOp>(
      yield.getOperand(0).getDefiningOp());
}

/// Recovers the chain `m_j = maximumf(m_{j-1}, rowmax_j)` that threads the
/// running maximum through the unrolled lanes, together with the value it
/// starts from.  Returns an empty vector when the body does not have that
/// shape, which is the signal to leave it alone.
SmallVector<Lane> findRunningMaxChain(Block *body, Value &chainInput) {
  SmallVector<Lane> lanes;
  Value running;
  for (Operation &op : *body) {
    auto maxOp = dyn_cast<arith::MaximumFOp>(&op);
    if (!maxOp)
      continue;
    // One operand must be a row max; the other continues the chain.
    auto lhsReduce = maxOp.getLhs().getDefiningOp<linalg::ReduceOp>();
    auto rhsReduce = maxOp.getRhs().getDefiningOp<linalg::ReduceOp>();
    linalg::ReduceOp reduce = (rhsReduce && isMaxReduce(rhsReduce)) ? rhsReduce
                              : (lhsReduce && isMaxReduce(lhsReduce))
                                  ? lhsReduce
                                  : nullptr;
    if (!reduce)
      continue;
    Value carried = reduce == rhsReduce ? maxOp.getLhs() : maxOp.getRhs();
    if (lanes.empty())
      chainInput = carried;
    else if (carried != running)
      return {}; // a second, unrelated chain: not the shape we handle
    running = maxOp.getResult();
    lanes.push_back({reduce, maxOp});
  }
  return lanes;
}

/// Rewrites `x * alpha` to `x` for every consumer of a rescale factor that is
/// now one, including the broadcast the accumulator's rescale goes through.
/// Returns false when a consumer is not one of those two shapes, in which case
/// nothing has been changed.
bool retireUnitRescale(Value alpha) {
  SmallVector<Operation *> toErase;
  for (Operation *user : llvm::make_early_inc_range(alpha.getUsers())) {
    if (auto mul = dyn_cast<arith::MulFOp>(user)) {
      Value other = mul.getLhs() == alpha ? mul.getRhs() : mul.getLhs();
      mul.getResult().replaceAllUsesWith(other);
      toErase.push_back(mul);
      continue;
    }
    if (auto bcast = dyn_cast<linalg::BroadcastOp>(user)) {
      for (Operation *bcastUser :
           llvm::make_early_inc_range(bcast->getResult(0).getUsers())) {
        auto mul = dyn_cast<arith::MulFOp>(bcastUser);
        if (!mul)
          return false;
        Value other =
            mul.getLhs() == bcast->getResult(0) ? mul.getRhs() : mul.getLhs();
        mul.getResult().replaceAllUsesWith(other);
        toErase.push_back(mul);
      }
      toErase.push_back(bcast);
      continue;
    }
    return false;
  }
  for (Operation *op : toErase)
    if (op->use_empty())
      op->erase();
  return true;
}

/// The `exp(m_prev - m_j)` feeding lane `j`'s rescales, or null.
math::ExpOp findRescale(Value prev, Value current) {
  for (Operation *user : current.getUsers()) {
    auto sub = dyn_cast<arith::SubFOp>(user);
    if (!sub || sub.getLhs() != prev || sub.getRhs() != current)
      continue;
    for (Operation *subUser : sub.getResult().getUsers())
      if (auto exp = dyn_cast<math::ExpOp>(subUser))
        return exp;
  }
  return nullptr;
}

/// Moves the accumulator fold to the end of the product chain.
///
/// After regrouping, lane 0's product still starts from `acc * alpha` while the
/// later lanes start from the previous product.  Unfusing that first init would
/// make lane 0's result VECTOR-produced and break the chain for lane 1, so the
/// whole chain is started from zero instead and the running accumulator is
/// folded in once, after the last product.  That is what leaves the chain
/// entirely on CUBE, with a single result crossing to VECTOR per group.
LogicalResult foldAccumulatorAfterChain(Block *body) {
  // The chain: matmuls whose init is either a VECTOR value or the previous
  // matmul's result, in program order.
  SmallVector<linalg::MatmulOp> chain;
  Value chainInit;
  for (Operation &op : *body) {
    auto matmul = dyn_cast<linalg::MatmulOp>(&op);
    if (!matmul || matmul.getDpsInits().size() != 1)
      continue;
    Value init = matmul.getDpsInits().front();
    if (!chain.empty() && init == chain.back()->getResult(0)) {
      chain.push_back(matmul);
      continue;
    }
    if (init.getDefiningOp<arith::MulFOp>()) {
      chain.assign({matmul});
      chainInit = init;
    }
  }
  if (chain.size() < 2)
    return success(); // nothing chained: leave the body as it is

  auto tensorType = dyn_cast<RankedTensorType>(chainInit.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return success();

  OpBuilder builder(chain.front());
  Location loc = chain.front().getLoc();
  Value zero = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(tensorType.getElementType()));
  Value empty = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                tensorType.getElementType());
  Value zeroed = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
  chain.front().setDpsInitOperand(0, zeroed);

  builder.setInsertionPointAfter(chain.back());
  Value product = chain.back()->getResult(0);
  auto folded = builder.create<arith::AddFOp>(loc, chainInit, product);
  product.replaceAllUsesExcept(folded.getResult(), folded);

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] softmax regroup: " << chain.size()
             << " products now chain from zero, with the accumulator folded in "
                "once after the last one\n");
  return success();
}

} // namespace

LogicalResult regroupSoftmaxMax(scf::ForOp loop, unsigned lanes) {
  if (lanes < 2)
    return success();

  Block *body = loop.getBody();
  Value chainInput;
  SmallVector<Lane> chain = findRunningMaxChain(body, chainInput);
  if (chain.size() != lanes) {
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] softmax regroup: found a chain of "
                            << chain.size() << " running maxima, expected "
                            << lanes << "; leaving the body alone\n");
    return success();
  }

  // The rescales that become one.  Collect them before anything moves.
  SmallVector<math::ExpOp> unitRescales;
  for (unsigned j = 1; j < lanes; ++j) {
    math::ExpOp exp = findRescale(chain[j - 1].running.getResult(),
                                  chain[j].running.getResult());
    if (!exp) {
      LLVM_DEBUG(llvm::dbgs() << "[cv-split] softmax regroup: lane " << j
                              << " has no recognisable rescale; bailing\n");
      return success();
    }
    unitRescales.push_back(exp);
  }

  // Everything downstream of an early lane's running max has to follow the
  // last row max, because it now reads a maximum taken over the whole group.
  Operation *anchor = chain.back().rowMax;
  SetVector<Operation *> toMove;
  for (unsigned j = 0; j + 1 < lanes; ++j) {
    SetVector<Operation *> slice;
    ForwardSliceOptions options;
    options.filter = [&](Operation *op) { return op->getBlock() == body; };
    getForwardSlice(chain[j].running.getResult(), &slice, options);
    for (Operation *op : slice)
      if (op->getBlock() == body && op->isBeforeInBlock(anchor))
        toMove.insert(op);
  }
  SmallVector<Operation *> ordered(toMove.begin(), toMove.end());
  llvm::sort(ordered,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  // Nothing in the slice may feed a later lane's row max, or moving it down
  // would break that lane's operands.
  for (Operation *op : ordered)
    for (Operation *user : op->getUsers())
      if (user->getBlock() == body && !toMove.contains(user) &&
          !user->isBeforeInBlock(anchor) && user != anchor)
        continue;

  Operation *insertAfter = anchor;
  for (Operation *op : ordered) {
    op->moveAfter(insertAfter);
    insertAfter = op;
  }

  // One maximum over the whole group, in place of the per-lane chain.
  OpBuilder builder(loop.getContext());
  builder.setInsertionPointAfter(anchor);
  Value groupMax = chainInput;
  for (Lane &lane : chain)
    groupMax = builder.create<arith::MaximumFOp>(
        lane.running.getLoc(), groupMax, lane.rowMax->getResult(0));
  for (Lane &lane : chain)
    lane.running.getResult().replaceAllUsesWith(groupMax);

  // With one shared maximum the later lanes' rescale factors are exp(0) = 1.
  // Retire their consumers, then the now-dead factor and the subtraction that
  // fed it, so the schedule does not carry work whose result is one.
  for (math::ExpOp exp : unitRescales)
    if (!retireUnitRescale(exp.getResult())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cv-split] softmax regroup: unexpected "
                    "rescale consumer; IR left partly regrouped\n");
      return failure();
    }

  for (math::ExpOp exp : unitRescales) {
    Operation *sub = exp.getOperand().getDefiningOp();
    if (exp->use_empty())
      exp->erase();
    if (sub && sub->use_empty())
      sub->erase();
  }

  for (Lane &lane : llvm::reverse(chain))
    if (lane.running->use_empty())
      lane.running->erase();

  LLVM_DEBUG(llvm::dbgs() << "[cv-split] softmax regroup: " << lanes
                          << " lanes now share one maximum; " << ordered.size()
                          << " operations moved after the last row max\n");

  return foldAccumulatorAfterChain(body);
}

} // namespace mlir::triton::cv_split
