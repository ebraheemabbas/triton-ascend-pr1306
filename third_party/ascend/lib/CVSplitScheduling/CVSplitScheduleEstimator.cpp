/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "CVSplitScheduleEstimator.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir::triton::cv_split::detail {
namespace {

constexpr unsigned kPrincipalResourceCount =
    static_cast<unsigned>(PrincipalResource::ScalarControl) + 1;
constexpr uint64_t kMaximumExpandedNodes = 65536;
constexpr uint32_t kScheduleUncertaintyBasisPoints = 500;

struct Reservation {
  uint64_t begin;
  uint64_t end;
};

struct IncomingEdge {
  unsigned predecessor;
  CVSplitEdgeKind kind;
};

struct ExpandedNode {
  unsigned iteration;
  unsigned requestOrder;
  uint32_t nodeId;
  const CVSplitPrimitiveEstimate *primitive;
  unsigned indegree = 0;
  llvm::SmallVector<unsigned> successors;
  llvm::SmallVector<IncomingEdge> predecessors;
};

static CVSplitScheduleEstimate makeFailure(uint32_t candidateId,
                                           CVSplitCandidateStatus status) {
  CVSplitScheduleEstimate estimate{};
  estimate.candidateId = candidateId;
  estimate.status = status;
  estimate.uncertaintyBasisPoints = 10000;
  return estimate;
}

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

static bool validResource(PrincipalResource resource) {
  unsigned index = static_cast<unsigned>(resource);
  return index < kPrincipalResourceCount;
}

static bool validEdgeKind(CVSplitEdgeKind kind) {
  switch (kind) {
  case CVSplitEdgeKind::DataDependency:
  case CVSplitEdgeKind::CrossCoreAvailability:
  case CVSplitEdgeKind::StorageReuse:
  case CVSplitEdgeKind::ResourceSerialization:
  case CVSplitEdgeKind::EventGenerateWait:
  case CVSplitEdgeKind::LoopCarried:
  case CVSplitEdgeKind::WaveOrTail:
    return true;
  }
  return false;
}

static bool validPrimitiveKind(CVSplitPrimitiveKind kind) {
  switch (kind) {
  case CVSplitPrimitiveKind::Cube:
  case CVSplitPrimitiveKind::VectorRegion:
  case CVSplitPrimitiveKind::Transfer:
  case CVSplitPrimitiveKind::Synchronization:
    return true;
  }
  return false;
}

static bool intervalsOverlap(uint64_t lhsBegin, uint64_t lhsEnd,
                             uint64_t rhsBegin, uint64_t rhsEnd) {
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

static CVSplitCandidateStatus
validatePrimitive(const CVSplitCostedNode &node,
                  CVSplitCalibrationId calibrationId) {
  if (!validPrimitiveKind(node.kind) ||
      node.primitive.status != CVSplitPrimitiveStatus::Success ||
      node.primitive.resultReadyCycles == 0 ||
      node.primitive.initiationIntervalCycles == 0 ||
      node.primitive.occupancy.empty())
    return CVSplitCandidateStatus::PrimitiveUnscoreable;
  if (node.primitive.calibrationId.value == 0 ||
      node.primitive.calibrationId.value != calibrationId.value)
    return CVSplitCandidateStatus::OutOfCalibration;

  for (auto [index, occupancy] : llvm::enumerate(node.primitive.occupancy)) {
    if (!validResource(occupancy.resource) ||
        occupancy.beginCycle >= occupancy.endCycle ||
        occupancy.endCycle > node.primitive.resultReadyCycles)
      return CVSplitCandidateStatus::InvalidRequest;
    for (const CVSplitResourceOccupancy &other :
         llvm::drop_begin(node.primitive.occupancy, index + 1))
      if (occupancy.resource == other.resource &&
          intervalsOverlap(occupancy.beginCycle, occupancy.endCycle,
                           other.beginCycle, other.endCycle))
        return CVSplitCandidateStatus::ResourceConflict;
  }
  return CVSplitCandidateStatus::Success;
}

static bool findResourceStart(
    const CVSplitPrimitiveEstimate &primitive, uint64_t dependencyReady,
    const std::array<llvm::SmallVector<Reservation>, kPrincipalResourceCount>
        &reservations,
    uint64_t &start) {
  start = dependencyReady;
  uint64_t reservationCount = 0;
  for (const auto &resourceReservations : reservations)
    reservationCount += resourceReservations.size();
  uint64_t maximumShifts;
  if (!checkedAdd(reservationCount, primitive.occupancy.size() + 1,
                  maximumShifts))
    return false;

  for (uint64_t shift = 0; shift <= maximumShifts; ++shift) {
    uint64_t nextStart = start;
    for (const CVSplitResourceOccupancy &occupancy : primitive.occupancy) {
      uint64_t absoluteBegin, absoluteEnd;
      if (!checkedAdd(start, occupancy.beginCycle, absoluteBegin) ||
          !checkedAdd(start, occupancy.endCycle, absoluteEnd))
        return false;
      const auto &resourceReservations =
          reservations[static_cast<unsigned>(occupancy.resource)];
      for (const Reservation &reservation : resourceReservations) {
        if (!intervalsOverlap(absoluteBegin, absoluteEnd, reservation.begin,
                              reservation.end))
          continue;
        if (reservation.end < occupancy.beginCycle)
          return false;
        nextStart = std::max(nextStart, reservation.end - occupancy.beginCycle);
      }
    }
    if (nextStart == start)
      return true;
    start = nextStart;
  }
  return false;
}

static bool appendReservations(
    const CVSplitPrimitiveEstimate &primitive, uint64_t start,
    std::array<llvm::SmallVector<Reservation>, kPrincipalResourceCount>
        &reservations,
    uint64_t &finish) {
  finish = start;
  if (!checkedAdd(start, primitive.resultReadyCycles, finish))
    return false;
  for (const CVSplitResourceOccupancy &occupancy : primitive.occupancy) {
    uint64_t begin, end;
    if (!checkedAdd(start, occupancy.beginCycle, begin) ||
        !checkedAdd(start, occupancy.endCycle, end))
      return false;
    reservations[static_cast<unsigned>(occupancy.resource)].push_back(
        {begin, end});
    finish = std::max(finish, end);
  }
  return true;
}

static bool buildResourceSummaries(
    const std::array<llvm::SmallVector<Reservation>, kPrincipalResourceCount>
        &reservations,
    uint64_t exposedWaitCycles,
    llvm::SmallVector<CVSplitResourceSummary> &summaries) {
  for (unsigned index = 0; index < kPrincipalResourceCount; ++index) {
    PrincipalResource resource = static_cast<PrincipalResource>(index);
    const auto &resourceReservations = reservations[index];
    CVSplitResourceSummary summary{resource, 0, 0, 0, 0, 0};
    if (!resourceReservations.empty()) {
      summary.firstUseCycle = std::numeric_limits<uint64_t>::max();
      for (const Reservation &reservation : resourceReservations) {
        uint64_t duration = reservation.end - reservation.begin;
        if (!checkedAdd(summary.busyCycles, duration, summary.busyCycles))
          return false;
        summary.firstUseCycle =
            std::min(summary.firstUseCycle, reservation.begin);
        summary.lastUseCycle = std::max(summary.lastUseCycle, reservation.end);
      }
      uint64_t window = summary.lastUseCycle - summary.firstUseCycle;
      if (summary.busyCycles > window)
        return false;
      summary.idleCycles = window - summary.busyCycles;
    }
    if (resource == PrincipalResource::ScalarControl)
      summary.blockedCycles = exposedWaitCycles;
    summaries.push_back(summary);
  }
  return true;
}

} // namespace

CVSplitScheduleEstimate
estimateDeterministicSchedule(CVSplitCalibrationId calibrationId,
                              const CVSplitScheduleEstimateRequest &request) {
  if (request.nodes.empty() || request.logicalUnrollFactor == 0 ||
      request.modeledIterations == 0 || calibrationId.value == 0)
    return makeFailure(request.candidateId,
                       CVSplitCandidateStatus::InvalidRequest);

  uint64_t expandedNodeCount;
  if (!checkedMul(request.nodes.size(), request.modeledIterations,
                  expandedNodeCount))
    return makeFailure(request.candidateId,
                       CVSplitCandidateStatus::ArithmeticOverflow);
  if (expandedNodeCount == 0 || expandedNodeCount > kMaximumExpandedNodes)
    return makeFailure(request.candidateId,
                       CVSplitCandidateStatus::InvalidRequest);

  llvm::DenseMap<uint32_t, unsigned> nodeOrderById;
  uint32_t maximumUncertainty = 0;
  for (auto [order, node] : llvm::enumerate(request.nodes)) {
    if (!nodeOrderById.try_emplace(node.nodeId, order).second)
      return makeFailure(request.candidateId,
                         CVSplitCandidateStatus::InvalidRequest);
    CVSplitCandidateStatus status = validatePrimitive(node, calibrationId);
    if (status != CVSplitCandidateStatus::Success)
      return makeFailure(request.candidateId, status);
    maximumUncertainty =
        std::max(maximumUncertainty, node.primitive.uncertaintyBasisPoints);
  }

  for (const CVSplitCostedEdge &edge : request.edges) {
    if (!validEdgeKind(edge.kind) || !nodeOrderById.contains(edge.fromNode) ||
        !nodeOrderById.contains(edge.toNode) ||
        (edge.iterationDistance == 0 && edge.fromNode == edge.toNode))
      return makeFailure(request.candidateId,
                         CVSplitCandidateStatus::InvalidRequest);
  }

  const unsigned nodesPerIteration = request.nodes.size();
  std::vector<ExpandedNode> expanded;
  expanded.reserve(expandedNodeCount);
  for (unsigned iteration = 0; iteration < request.modeledIterations;
       ++iteration)
    for (auto [order, node] : llvm::enumerate(request.nodes))
      expanded.push_back({iteration, static_cast<unsigned>(order), node.nodeId,
                          &node.primitive});

  for (const CVSplitCostedEdge &edge : request.edges) {
    unsigned fromOrder = nodeOrderById.lookup(edge.fromNode);
    unsigned toOrder = nodeOrderById.lookup(edge.toNode);
    for (unsigned iteration = 0; iteration < request.modeledIterations;
         ++iteration) {
      uint64_t targetIteration;
      if (!checkedAdd(iteration, edge.iterationDistance, targetIteration))
        return makeFailure(request.candidateId,
                           CVSplitCandidateStatus::ArithmeticOverflow);
      if (targetIteration >= request.modeledIterations)
        continue;
      unsigned sourceIndex = iteration * nodesPerIteration + fromOrder;
      unsigned targetIndex =
          static_cast<unsigned>(targetIteration) * nodesPerIteration + toOrder;
      expanded[sourceIndex].successors.push_back(targetIndex);
      expanded[targetIndex].predecessors.push_back({sourceIndex, edge.kind});
      ++expanded[targetIndex].indegree;
    }
  }

  auto laterReadyNode = [&](unsigned lhs, unsigned rhs) {
    const ExpandedNode &left = expanded[lhs];
    const ExpandedNode &right = expanded[rhs];
    return std::tie(left.iteration, left.requestOrder, left.nodeId) >
           std::tie(right.iteration, right.requestOrder, right.nodeId);
  };
  std::priority_queue<unsigned, std::vector<unsigned>, decltype(laterReadyNode)>
      ready(laterReadyNode);
  for (unsigned index = 0; index < expanded.size(); ++index)
    if (expanded[index].indegree == 0)
      ready.push(index);

  std::array<llvm::SmallVector<Reservation>, kPrincipalResourceCount>
      reservations;
  std::vector<uint64_t> starts(expanded.size(), 0);
  std::vector<uint64_t> resultReady(expanded.size(), 0);
  uint64_t criticalPath = 0;
  uint64_t exposedWaitCycles = 0;
  unsigned scheduled = 0;

  while (!ready.empty()) {
    unsigned index = ready.top();
    ready.pop();
    ExpandedNode &node = expanded[index];
    uint64_t dependencyReady = 0;
    uint64_t nonEventReady = 0;
    bool hasEventDependency = false;
    for (const IncomingEdge &incoming : node.predecessors) {
      dependencyReady =
          std::max(dependencyReady, resultReady[incoming.predecessor]);
      if (incoming.kind == CVSplitEdgeKind::EventGenerateWait)
        hasEventDependency = true;
      else
        nonEventReady =
            std::max(nonEventReady, resultReady[incoming.predecessor]);
    }
    if (hasEventDependency && dependencyReady > nonEventReady) {
      uint64_t exposed = dependencyReady - nonEventReady;
      if (!checkedAdd(exposedWaitCycles, exposed, exposedWaitCycles))
        return makeFailure(request.candidateId,
                           CVSplitCandidateStatus::ArithmeticOverflow);
    }

    uint64_t start;
    if (!findResourceStart(*node.primitive, dependencyReady, reservations,
                           start))
      return makeFailure(request.candidateId,
                         CVSplitCandidateStatus::ResourceConflict);
    uint64_t finish;
    if (!appendReservations(*node.primitive, start, reservations, finish))
      return makeFailure(request.candidateId,
                         CVSplitCandidateStatus::ArithmeticOverflow);
    uint64_t readyCycle;
    if (!checkedAdd(start, node.primitive->resultReadyCycles, readyCycle))
      return makeFailure(request.candidateId,
                         CVSplitCandidateStatus::ArithmeticOverflow);
    starts[index] = start;
    resultReady[index] = readyCycle;
    criticalPath = std::max(criticalPath, finish);
    ++scheduled;

    for (unsigned successor : node.successors) {
      if (expanded[successor].indegree == 0)
        return makeFailure(request.candidateId,
                           CVSplitCandidateStatus::InvalidRequest);
      if (--expanded[successor].indegree == 0)
        ready.push(successor);
    }
  }

  if (scheduled != expanded.size())
    return makeFailure(request.candidateId,
                       CVSplitCandidateStatus::InvalidRequest);

  std::vector<uint64_t> iterationAnchors(request.modeledIterations, 0);
  for (unsigned iteration = 0; iteration < request.modeledIterations;
       ++iteration)
    iterationAnchors[iteration] = starts[iteration * nodesPerIteration];
  uint64_t steadyStateII = criticalPath;
  if (iterationAnchors.size() > 1) {
    steadyStateII = 0;
    for (unsigned iteration = 1; iteration < iterationAnchors.size();
         ++iteration) {
      if (iterationAnchors[iteration] < iterationAnchors[iteration - 1])
        return makeFailure(request.candidateId,
                           CVSplitCandidateStatus::InvalidRequest);
      steadyStateII =
          std::max(steadyStateII, iterationAnchors[iteration] -
                                      iterationAnchors[iteration - 1]);
    }
  }

  uint64_t uncertainty;
  if (!checkedAdd(maximumUncertainty, kScheduleUncertaintyBasisPoints,
                  uncertainty))
    return makeFailure(request.candidateId,
                       CVSplitCandidateStatus::ArithmeticOverflow);

  CVSplitScheduleEstimate estimate{};
  estimate.candidateId = request.candidateId;
  estimate.status = CVSplitCandidateStatus::Success;
  estimate.prologueCycles = iterationAnchors.front();
  estimate.steadyStateInitiationIntervalCycles = steadyStateII;
  estimate.criticalPathCycles = criticalPath;
  estimate.epilogueCycles = criticalPath - iterationAnchors.back();
  estimate.exposedWaitCycles = exposedWaitCycles;
  estimate.uncertaintyBasisPoints =
      static_cast<uint32_t>(std::min<uint64_t>(uncertainty, 10000));
  if (!buildResourceSummaries(reservations, exposedWaitCycles,
                              estimate.resources))
    return makeFailure(request.candidateId,
                       CVSplitCandidateStatus::ArithmeticOverflow);
  return estimate;
}

} // namespace mlir::triton::cv_split::detail
