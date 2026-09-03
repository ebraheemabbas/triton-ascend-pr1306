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

#include "llvm/Support/Error.h"

#include <utility>

namespace mlir::triton::cv_split {
namespace {

constexpr uint32_t kExperimentalA5ProductId = 9579;
constexpr uint32_t kExperimentalA5Revision = 0;
constexpr uint32_t kExperimentalCalibrationSchemaVersion = 1;
constexpr CVSplitCalibrationId kExperimentalCalibrationId{
    0xA500000000000801ULL};

static bool sameTarget(CVSplitTargetIdentity lhs, CVSplitTargetIdentity rhs) {
  return lhs.archFamily == rhs.archFamily && lhs.productId == rhs.productId &&
         lhs.revision == rhs.revision;
}

static CVSplitPrimitiveEstimate
makePrimitiveFailure(CVSplitPrimitiveStatus status,
                     CVSplitCalibrationId calibrationId) {
  CVSplitPrimitiveEstimate estimate{};
  estimate.status = status;
  estimate.uncertaintyBasisPoints = 10000;
  estimate.calibrationId = calibrationId;
  return estimate;
}

} // namespace

struct CVSplitPrimitiveCostModel::Impl {
  explicit Impl(CVSplitCostModelInfo modelInfo) : info(modelInfo) {}

  CVSplitCostModelInfo info;
};

llvm::Expected<std::unique_ptr<CVSplitPrimitiveCostModel>>
CVSplitPrimitiveCostModel::createForTarget(CVSplitTargetIdentity target) {
  if (target.archFamily != CVSplitArchFamily::A5 ||
      target.productId != kExperimentalA5ProductId ||
      target.revision != kExperimentalA5Revision) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "unsupported CVSplit cost-model target: arch=%u product=%u "
        "revision=%u",
        static_cast<unsigned>(target.archFamily), target.productId,
        target.revision);
  }

  CVSplitCostModelInfo info{kCVSplitCostModelApiVersion,
                            kExperimentalCalibrationSchemaVersion,
                            kExperimentalCalibrationId, target};
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
  return makePrimitiveFailure(sameTarget(request.target, impl_->info.target)
                                  ? CVSplitPrimitiveStatus::MissingCalibration
                                  : CVSplitPrimitiveStatus::TargetUnsupported,
                              impl_->info.calibrationId);
}

CVSplitPrimitiveEstimate CVSplitPrimitiveCostModel::estimateVectorRegion(
    const CVSplitVectorRegionRequest &request) const {
  return makePrimitiveFailure(sameTarget(request.target, impl_->info.target)
                                  ? CVSplitPrimitiveStatus::MissingCalibration
                                  : CVSplitPrimitiveStatus::TargetUnsupported,
                              impl_->info.calibrationId);
}

CVSplitPrimitiveEstimate CVSplitPrimitiveCostModel::estimateTransfer(
    const CVSplitTransferRequest &request) const {
  return makePrimitiveFailure(sameTarget(request.target, impl_->info.target)
                                  ? CVSplitPrimitiveStatus::MissingCalibration
                                  : CVSplitPrimitiveStatus::TargetUnsupported,
                              impl_->info.calibrationId);
}

CVSplitPrimitiveEstimate CVSplitPrimitiveCostModel::estimateSynchronization(
    const CVSplitSynchronizationRequest &request) const {
  return makePrimitiveFailure(sameTarget(request.target, impl_->info.target)
                                  ? CVSplitPrimitiveStatus::MissingCalibration
                                  : CVSplitPrimitiveStatus::TargetUnsupported,
                              impl_->info.calibrationId);
}

CVSplitScheduleEstimate CVSplitPrimitiveCostModel::estimateSchedule(
    const CVSplitScheduleEstimateRequest &request) const {
  CVSplitScheduleEstimate estimate{};
  estimate.candidateId = request.candidateId;
  estimate.status = CVSplitCandidateStatus::PrimitiveUnscoreable;
  estimate.uncertaintyBasisPoints = 10000;
  return estimate;
}

} // namespace mlir::triton::cv_split
