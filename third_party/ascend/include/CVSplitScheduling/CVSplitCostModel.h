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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CV_SPLIT_COST_MODEL_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CV_SPLIT_COST_MODEL_H

#include "ascend/include/CVSplitScheduling/CVSplitCostModelTypes.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::triton::cv_split {

/// Immutable target-specific primitive and schedule estimator.
///
/// Stage 8.1 installs only the stable v4 façade and target/calibration
/// identity. Primitive and schedule queries deliberately return explicit
/// unscoreable statuses until their versioned implementation stages land.
class CVSplitPrimitiveCostModel final {
public:
  static llvm::Expected<std::unique_ptr<CVSplitPrimitiveCostModel>>
  createForTarget(CVSplitTargetIdentity target);

  ~CVSplitPrimitiveCostModel();

  CVSplitPrimitiveCostModel(const CVSplitPrimitiveCostModel &) = delete;
  CVSplitPrimitiveCostModel &
  operator=(const CVSplitPrimitiveCostModel &) = delete;
  CVSplitPrimitiveCostModel(CVSplitPrimitiveCostModel &&) = delete;
  CVSplitPrimitiveCostModel &operator=(CVSplitPrimitiveCostModel &&) = delete;

  CVSplitCostModelInfo getInfo() const;

  CVSplitPrimitiveEstimate estimateCube(const CVSplitCubeRequest &) const;
  CVSplitPrimitiveEstimate
  estimateVectorRegion(const CVSplitVectorRegionRequest &) const;
  CVSplitPrimitiveEstimate
  estimateTransfer(const CVSplitTransferRequest &) const;
  CVSplitPrimitiveEstimate
  estimateSynchronization(const CVSplitSynchronizationRequest &) const;

  CVSplitScheduleEstimate
  estimateSchedule(const CVSplitScheduleEstimateRequest &) const;

private:
  struct Impl;
  explicit CVSplitPrimitiveCostModel(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> impl_;
};

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CV_SPLIT_COST_MODEL_H
