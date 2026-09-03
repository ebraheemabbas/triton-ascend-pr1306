/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include "ascend/include/CVSplitScheduling/CostModelCandidateRanking.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace mlir::triton::cv_split {
namespace {

#include "CVSplitRankingA5Experimental.inc"

static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool checkedCeilDiv(uint64_t numerator, uint64_t denominator,
                           uint64_t &result) {
  if (denominator == 0)
    return false;
  result = numerator / denominator + (numerator % denominator != 0);
  return true;
}

static bool structurallyDominates(const CVSplitCandidateRankEntry &lhs,
                                  const CVSplitCandidateRankEntry &rhs) {
  if (lhs.matrixLineageInFlightLimits.size() !=
      rhs.matrixLineageInFlightLimits.size())
    return false;
  bool strictlyShallower = false;
  for (auto [left, right] : llvm::zip(lhs.matrixLineageInFlightLimits,
                                      rhs.matrixLineageInFlightLimits)) {
    if (left > right)
      return false;
    strictlyShallower |= left < right;
  }
  return strictlyShallower;
}

static CVSplitCandidateRanking
makeFailure(CVSplitCandidateRankingStatus status,
            std::optional<uint32_t> fallbackCandidateId) {
  CVSplitCandidateRanking result;
  result.status = status;
  result.fallbackCandidateId = fallbackCandidateId;
  return result;
}

} // namespace

CVSplitCandidateRanking rankCostModelCandidates(
    llvm::ArrayRef<CVSplitOwnedScheduleRequest> graphs,
    llvm::ArrayRef<CVSplitScheduleEstimate> estimates,
    std::optional<uint32_t> fallbackCandidateId) {
  if (graphs.empty() || graphs.size() != estimates.size())
    return makeFailure(CVSplitCandidateRankingStatus::InvalidInput,
                       fallbackCandidateId);

  CVSplitCandidateRanking result;
  result.status = CVSplitCandidateRankingStatus::Success;
  result.fallbackCandidateId = fallbackCandidateId;
  llvm::DenseSet<uint32_t> candidateIds;
  bool foundFallback = !fallbackCandidateId.has_value();

  for (auto [graph, estimate] : llvm::zip(graphs, estimates)) {
    if (graph.candidateId != estimate.candidateId ||
        !candidateIds.insert(graph.candidateId).second ||
        graph.matrixLineageInFlightLimits.empty() ||
        graph.matrixLineageInFlightLimits.size() !=
            graph.matrixLineageResultBytes.size())
      return makeFailure(CVSplitCandidateRankingStatus::InvalidInput,
                         fallbackCandidateId);
    if (estimate.status != CVSplitCandidateStatus::Success ||
        estimate.steadyStateInitiationIntervalCycles == 0)
      return makeFailure(CVSplitCandidateRankingStatus::Unscoreable,
                         fallbackCandidateId);

    uint64_t extraLiveBytes = 0;
    for (auto [depth, bytes] :
         llvm::zip(graph.matrixLineageInFlightLimits,
                   graph.matrixLineageResultBytes)) {
      if (depth == 0 || bytes == 0)
        return makeFailure(CVSplitCandidateRankingStatus::InvalidInput,
                           fallbackCandidateId);
      uint64_t lineageBytes;
      if (!checkedMul(static_cast<uint64_t>(depth - 1), bytes,
                      lineageBytes) ||
          !checkedAdd(extraLiveBytes, lineageBytes, extraLiveBytes))
        return makeFailure(CVSplitCandidateRankingStatus::ArithmeticOverflow,
                           fallbackCandidateId);
    }
    uint64_t pressureCycles;
    if (!checkedCeilDiv(extraLiveBytes,
                        kExperimentalLiveResultPressureBytesPerCycle,
                        pressureCycles))
      return makeFailure(CVSplitCandidateRankingStatus::ArithmeticOverflow,
                         fallbackCandidateId);
    uint64_t policyScore;
    if (!checkedAdd(estimate.steadyStateInitiationIntervalCycles,
                    pressureCycles, policyScore))
      return makeFailure(CVSplitCandidateRankingStatus::ArithmeticOverflow,
                         fallbackCandidateId);

    CVSplitCandidateRankEntry entry;
    entry.candidateId = graph.candidateId;
    entry.rawInitiationIntervalCycles =
        estimate.steadyStateInitiationIntervalCycles;
    entry.liveResultPressureCycles = pressureCycles;
    entry.policyScoreCycles = policyScore;
    entry.uncertaintyBasisPoints = estimate.uncertaintyBasisPoints;
    entry.matrixLineageInFlightLimits = graph.matrixLineageInFlightLimits;
    result.entries.push_back(std::move(entry));
    foundFallback |= fallbackCandidateId == graph.candidateId;
  }
  if (!foundFallback)
    return makeFailure(CVSplitCandidateRankingStatus::InvalidInput,
                       fallbackCandidateId);

  llvm::sort(result.entries,
             [](const CVSplitCandidateRankEntry &lhs,
                const CVSplitCandidateRankEntry &rhs) {
               if (lhs.policyScoreCycles != rhs.policyScoreCycles)
                 return lhs.policyScoreCycles < rhs.policyScoreCycles;
               bool lhsDominates = structurallyDominates(lhs, rhs);
               bool rhsDominates = structurallyDominates(rhs, lhs);
               if (lhsDominates != rhsDominates)
                 return lhsDominates;
               return lhs.candidateId < rhs.candidateId;
             });

  const CVSplitCandidateRankEntry &winner = result.entries.front();
  for (const CVSplitCandidateRankEntry &entry :
       llvm::drop_begin(result.entries)) {
    if (entry.policyScoreCycles != winner.policyScoreCycles)
      break;
    if (!structurallyDominates(winner, entry)) {
      result.status = CVSplitCandidateRankingStatus::Ambiguous;
      return result;
    }
  }

  result.predictedCandidateId = winner.candidateId;
  if (result.entries.size() == 1)
    return result;

  const CVSplitCandidateRankEntry &runnerUp = result.entries[1];
  result.runnerUpCandidateId = runnerUp.candidateId;
  result.marginCycles = runnerUp.policyScoreCycles - winner.policyScoreCycles;
  uint64_t scaledMargin;
  if (!checkedMul(result.marginCycles, 10000, scaledMargin))
    return makeFailure(CVSplitCandidateRankingStatus::ArithmeticOverflow,
                       fallbackCandidateId);
  result.marginBasisPoints = static_cast<uint32_t>(std::min<uint64_t>(
      scaledMargin / runnerUp.policyScoreCycles, 10000));
  uint64_t requiredMargin;
  if (!checkedAdd(std::max(winner.uncertaintyBasisPoints,
                           runnerUp.uncertaintyBasisPoints),
                  kExperimentalMinimumRankingMarginBasisPoints,
                  requiredMargin))
    return makeFailure(CVSplitCandidateRankingStatus::ArithmeticOverflow,
                       fallbackCandidateId);
  result.requiredMarginBasisPoints = static_cast<uint32_t>(
      std::min<uint64_t>(requiredMargin, 10000));
  result.clearsUncertainty =
      result.marginBasisPoints > result.requiredMarginBasisPoints;
  return result;
}

} // namespace mlir::triton::cv_split
