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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_SOFTMAX_REGROUP_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_SOFTMAX_REGROUP_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::cv_split {

/// Gives the unrolled lanes of a streaming softmax one shared maximum.
///
/// The streaming form takes a running maximum after every block, so each lane
/// carries its own rescale factor and each lane's product has to come back to
/// VECTOR to be folded into the accumulator.  Taking the maximum over all
/// `lanes` blocks first puts their P tiles on a common scale: the later lanes'
/// rescale factors become one, the accumulator is rescaled once for the group,
/// and the products accumulate into each other -- which is what lets them stay
/// in L0C instead of being drained per lane.
///
/// Same arithmetic, different association.  The group maximum is at least every
/// block maximum, so each exponent stays <= 0 and the numerics are no worse
/// than the streaming form.
///
/// Leaves the body untouched and returns success when it does not have the
/// expected shape, so a kernel that is not a streaming softmax is unaffected.
LogicalResult regroupSoftmaxMax(scf::ForOp loop, unsigned lanes);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_SOFTMAX_REGROUP_H
