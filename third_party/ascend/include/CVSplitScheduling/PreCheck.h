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

#ifndef TRITON_CV_SPLIT_SCHEDULING_PRE_CHECK_H
#define TRITON_CV_SPLIT_SCHEDULING_PRE_CHECK_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::cv_split {

struct AccumulatorJoinRewriteResult;

/// Checks the structural assumptions that must hold before CV split scheduling
/// mutates `funcOp`. Returns the single supported candidate loop on success.
FailureOr<scf::ForOp> preCheckCVSplitScheduling(func::FuncOp funcOp,
                                                int unrollFactor);

/// Record uses of the selected loop's iteration arguments before unrolling.
/// The temporary operand markers survive cloning, including a full unroll.
void markLoopCarriedConsumers(scf::ForOp loop);

/// After accumulator separation, reject cross-core SSA loop-carried edges,
/// including marked edges exposed inside the unrolled body. Same-iteration
/// cross-core transfers and same-core recurrences remain legal. This does not
/// replace the separate memory ownership/alias checks.
LogicalResult preCheckCrossIterationConsumers(
    scf::ForOp loop, const Classification &classification,
    const AccumulatorJoinRewriteResult &accumulatorJoins);

void removeLoopCarriedConsumerMarkers(Operation *root);

} // namespace mlir::triton::cv_split

#endif // TRITON_CV_SPLIT_SCHEDULING_PRE_CHECK_H
