/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_ASCEND_CV_SPLIT_EXPERIMENTAL_PRIMITIVE_COSTS_H
#define TRITON_ASCEND_CV_SPLIT_EXPERIMENTAL_PRIMITIVE_COSTS_H

#include "ascend/include/CVSplitScheduling/CVSplitCostModelTypes.h"

namespace mlir::triton::cv_split::detail {

bool isExperimentalA5Target(CVSplitTargetIdentity target);
CVSplitCostModelInfo getExperimentalA5ModelInfo(CVSplitTargetIdentity target);

CVSplitPrimitiveEstimate
estimateExperimentalCube(CVSplitTargetIdentity modelTarget,
                         CVSplitCalibrationId calibrationId,
                         const CVSplitCubeRequest &request);

CVSplitPrimitiveEstimate
estimateExperimentalVectorRegion(CVSplitTargetIdentity modelTarget,
                                 CVSplitCalibrationId calibrationId,
                                 const CVSplitVectorRegionRequest &request);

CVSplitPrimitiveEstimate
estimateExperimentalTransfer(CVSplitTargetIdentity modelTarget,
                             CVSplitCalibrationId calibrationId,
                             const CVSplitTransferRequest &request);

CVSplitPrimitiveEstimate estimateExperimentalSynchronization(
    CVSplitTargetIdentity modelTarget, CVSplitCalibrationId calibrationId,
    const CVSplitSynchronizationRequest &request);

} // namespace mlir::triton::cv_split::detail

#endif // TRITON_ASCEND_CV_SPLIT_EXPERIMENTAL_PRIMITIVE_COSTS_H
