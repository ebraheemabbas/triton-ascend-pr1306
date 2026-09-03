/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULE_ESTIMATOR_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULE_ESTIMATOR_H

#include "ascend/include/CVSplitScheduling/CVSplitCostModelTypes.h"

namespace mlir::triton::cv_split::detail {

CVSplitScheduleEstimate
estimateDeterministicSchedule(CVSplitCalibrationId calibrationId,
                              const CVSplitScheduleEstimateRequest &request);

} // namespace mlir::triton::cv_split::detail

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULE_ESTIMATOR_H
