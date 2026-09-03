/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_DIAGNOSTICS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_DIAGNOSTICS_H

#include "ascend/include/CVSplitScheduling/CostModelRequestExtraction.h"

#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::triton::cv_split {

/// Queries and logs every extracted primitive without estimating schedules or
/// selecting/mutating a candidate.
LogicalResult
logPrimitiveCostEstimates(const CVSplitCostModelRequestSet &requests);

LogicalResult
logCandidateScheduleEstimates(const CVSplitCostModelRequestSet &requests,
                              std::optional<unsigned> fallbackCandidateId);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_DIAGNOSTICS_H
