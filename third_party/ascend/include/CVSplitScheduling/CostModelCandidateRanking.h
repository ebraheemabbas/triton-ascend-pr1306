/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_CANDIDATE_RANKING_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_CANDIDATE_RANKING_H

#include "ascend/include/CVSplitScheduling/CostModelCandidateGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::triton::cv_split {

enum class CVSplitCandidateRankingStatus {
  Success,
  InvalidInput,
  Unscoreable,
  ArithmeticOverflow,
  Ambiguous,
};

struct CVSplitCandidateRankEntry {
  uint32_t candidateId = 0;
  uint64_t rawInitiationIntervalCycles = 0;
  uint64_t liveResultPressureCycles = 0;
  uint64_t policyScoreCycles = 0;
  uint32_t uncertaintyBasisPoints = 0;
  llvm::SmallVector<uint32_t> matrixLineageInFlightLimits;
};

struct CVSplitCandidateRanking {
  CVSplitCandidateRankingStatus status =
      CVSplitCandidateRankingStatus::InvalidInput;
  llvm::SmallVector<CVSplitCandidateRankEntry> entries;
  std::optional<uint32_t> predictedCandidateId;
  std::optional<uint32_t> runnerUpCandidateId;
  std::optional<uint32_t> fallbackCandidateId;
  uint64_t marginCycles = 0;
  uint32_t marginBasisPoints = 0;
  uint32_t requiredMarginBasisPoints = 0;
  bool clearsUncertainty = false;
};

/// Rank already legal structural candidates without mutating or selecting IR.
///
/// The target-specific pressure term is provisional. Candidate identity is
/// used only to join graph/estimate records and stabilize diagnostics.
CVSplitCandidateRanking rankCostModelCandidates(
    llvm::ArrayRef<CVSplitOwnedScheduleRequest> graphs,
    llvm::ArrayRef<CVSplitScheduleEstimate> estimates,
    std::optional<uint32_t> fallbackCandidateId);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_COST_MODEL_CANDIDATE_RANKING_H
