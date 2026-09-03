/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_CANDIDATE_GRAPH_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_CANDIDATE_GRAPH_H

#include "ascend/include/CVSplitScheduling/CVSplitCostModel.h"
#include "ascend/include/CVSplitScheduling/CostModelRequestExtraction.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::cv_split {

struct CVSplitOwnedScheduleRequest {
  uint32_t candidateId = 0;
  llvm::SmallVector<CVSplitCostedNode> nodes;
  llvm::SmallVector<CVSplitCostedEdge> edges;
  llvm::SmallVector<uint32_t> matrixLineageInFlightLimits;
  llvm::SmallVector<uint64_t> matrixLineageResultBytes;
  uint32_t logicalUnrollFactor = 0;
  uint32_t modeledIterations = 0;

  CVSplitScheduleEstimateRequest getRequest() const;
};

FailureOr<llvm::SmallVector<CVSplitOwnedScheduleRequest>>
buildCostModelCandidateGraphs(const CVSplitCostModelRequestSet &requests,
                              const CVSplitPrimitiveCostModel &model);

} // namespace mlir::triton::cv_split
#endif
