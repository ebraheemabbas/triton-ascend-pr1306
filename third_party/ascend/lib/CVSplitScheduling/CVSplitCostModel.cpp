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

#include "ascend/include/CVSplitScheduling/CVSplitCostModel.h"
#include "CVSplitExperimentalPrimitiveCosts.h"
#include "CVSplitScheduleEstimator.h"

#include "llvm/Support/Error.h"

#include <utility>

namespace mlir::triton::cv_split {

struct CVSplitPrimitiveCostModel::Impl {
  explicit Impl(CVSplitCostModelInfo modelInfo) : info(modelInfo) {}

  CVSplitCostModelInfo info;
};

llvm::Expected<std::unique_ptr<CVSplitPrimitiveCostModel>>
CVSplitPrimitiveCostModel::createForTarget(CVSplitTargetIdentity target) {
  if (!detail::isExperimentalA5Target(target)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "unsupported CVSplit cost-model target: arch=%u product=%u "
        "revision=%u",
        static_cast<unsigned>(target.archFamily), target.productId,
        target.revision);
  }

  CVSplitCostModelInfo info = detail::getExperimentalA5ModelInfo(target);
  auto implementation = std::make_unique<Impl>(info);
  return std::unique_ptr<CVSplitPrimitiveCostModel>(
      new CVSplitPrimitiveCostModel(std::move(implementation)));
}

CVSplitPrimitiveCostModel::CVSplitPrimitiveCostModel(
    std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}

CVSplitPrimitiveCostModel::~CVSplitPrimitiveCostModel() = default;

CVSplitCostModelInfo CVSplitPrimitiveCostModel::getInfo() const {
  return impl_->info;
}

CVSplitPrimitiveEstimate CVSplitPrimitiveCostModel::estimateCube(
    const CVSplitCubeRequest &request) const {
  return detail::estimateExperimentalCube(impl_->info.target,
                                          impl_->info.calibrationId, request);
}

CVSplitPrimitiveEstimate CVSplitPrimitiveCostModel::estimateVectorRegion(
    const CVSplitVectorRegionRequest &request) const {
  return detail::estimateExperimentalVectorRegion(
      impl_->info.target, impl_->info.calibrationId, request);
}

CVSplitPrimitiveEstimate CVSplitPrimitiveCostModel::estimateTransfer(
    const CVSplitTransferRequest &request) const {
  return detail::estimateExperimentalTransfer(
      impl_->info.target, impl_->info.calibrationId, request);
}

CVSplitPrimitiveEstimate CVSplitPrimitiveCostModel::estimateSynchronization(
    const CVSplitSynchronizationRequest &request) const {
  return detail::estimateExperimentalSynchronization(
      impl_->info.target, impl_->info.calibrationId, request);
}

CVSplitScheduleEstimate CVSplitPrimitiveCostModel::estimateSchedule(
    const CVSplitScheduleEstimateRequest &request) const {
  return detail::estimateDeterministicSchedule(impl_->info.calibrationId,
                                               request);
}

} // namespace mlir::triton::cv_split
