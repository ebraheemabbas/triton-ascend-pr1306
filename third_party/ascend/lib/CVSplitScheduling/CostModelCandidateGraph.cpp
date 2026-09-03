/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include "ascend/include/CVSplitScheduling/CostModelCandidateGraph.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::triton::cv_split {
namespace {
constexpr uint32_t kModeledOuterIterations = 3;

struct PhaseNodes {
  bool cubeToVector = false;
  uint32_t first = 0;
  uint32_t tail = 0;
  uint32_t cube = 0;
  uint32_t transfer = 0;
  uint32_t vector = 0;
};

static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool ceilDiv(uint64_t numerator, uint64_t denominator,
                    uint64_t &result) {
  if (denominator == 0)
    return false;
  result = numerator / denominator + (numerator % denominator != 0);
  return true;
}

static FailureOr<CVSplitPrimitiveEstimate>
buildVectorPhaseEstimate(const CVSplitCostModelRequestSet &requests,
                         const CVSplitPrimitiveCostModel &model,
                         unsigned phaseCount) {
  if (phaseCount == 0 || requests.vectorRegionRequests.empty())
    return failure();
  uint64_t totalReady = 0, totalRead = 0, totalWritten = 0;
  uint32_t uncertainty = 0;
  CVSplitCalibrationId calibration{0};
  for (const CVSplitOwnedVectorRegionRequest &owned :
       requests.vectorRegionRequests) {
    CVSplitPrimitiveEstimate estimate =
        model.estimateVectorRegion(owned.getRequest());
    if (estimate.status != CVSplitPrimitiveStatus::Success ||
        estimate.calibrationId.value == 0)
      return failure();
    if (calibration.value == 0)
      calibration = estimate.calibrationId;
    if (calibration.value != estimate.calibrationId.value ||
        !checkedAdd(totalReady, estimate.resultReadyCycles, totalReady) ||
        !checkedAdd(totalRead, estimate.bytesRead, totalRead) ||
        !checkedAdd(totalWritten, estimate.bytesWritten, totalWritten))
      return failure();
    uncertainty = std::max(uncertainty, estimate.uncertaintyBasisPoints);
  }
  uint64_t ready, bytesRead, bytesWritten;
  if (!ceilDiv(totalReady, phaseCount, ready) ||
      !ceilDiv(totalRead, phaseCount, bytesRead) ||
      !ceilDiv(totalWritten, phaseCount, bytesWritten))
    return failure();
  ready = std::max<uint64_t>(ready, 1);
  CVSplitPrimitiveEstimate phase{};
  phase.status = CVSplitPrimitiveStatus::Success;
  phase.resultReadyCycles = ready;
  phase.initiationIntervalCycles = ready;
  phase.occupancy.push_back({PrincipalResource::Vector, 0, ready});
  phase.bytesRead = bytesRead;
  phase.bytesWritten = bytesWritten;
  phase.uncertaintyBasisPoints = uncertainty;
  phase.calibrationId = calibration;
  return phase;
}

static FailureOr<CVSplitOwnedScheduleRequest>
buildOneGraph(const CVSplitCostModelRequestSet &requests,
              const CVSplitPrimitiveCostModel &model,
              const CVSplitExtractedCandidateSummary &candidate) {
  unsigned lanes = candidate.logicalLaneCount;
  if (lanes == 0 || candidate.waveWidth == 0 ||
      requests.transferRequests.empty() ||
      requests.transferRequests.size() % lanes != 0)
    return failure();
  unsigned lineageCount = requests.transferRequests.size() / lanes;
  if (lineageCount == 0 || requests.synchronizationRequests.size() <
                               2 * requests.transferRequests.size())
    return failure();

  llvm::SmallVector<unsigned> c2vPhases;
  for (unsigned phase = 0; phase < lineageCount; ++phase) {
    CVSplitTransferKind expected = requests.transferRequests[phase].kind;
    for (unsigned lane = 1; lane < lanes; ++lane)
      if (requests.transferRequests[lane * lineageCount + phase].kind !=
          expected)
        return failure();
    if (expected == CVSplitTransferKind::FixpipeDrain)
      c2vPhases.push_back(phase);
    else if (expected != CVSplitTransferKind::CopyAndLayoutConversion)
      return failure();
  }
  if (c2vPhases.empty() ||
      candidate.matrixLineages.size() != c2vPhases.size() ||
      requests.cubeRequests.size() != c2vPhases.size() * lanes)
    return failure();

  FailureOr<CVSplitPrimitiveEstimate> vectorPhase =
      buildVectorPhaseEstimate(requests, model, c2vPhases.size() * lanes);
  if (failed(vectorPhase))
    return failure();

  CVSplitOwnedScheduleRequest graph;
  graph.candidateId = candidate.candidateId;
  graph.logicalUnrollFactor = lanes;
  graph.modeledIterations = kModeledOuterIterations;
  graph.matrixLineageInFlightLimits.resize(c2vPhases.size());
  graph.matrixLineageResultBytes.resize(c2vPhases.size());
  std::vector<std::vector<PhaseNodes>> phaseNodes(
      lineageCount, std::vector<PhaseNodes>(lanes));
  auto addNode = [&](CVSplitPrimitiveKind kind,
                     CVSplitPrimitiveEstimate estimate) {
    uint32_t id = graph.nodes.size();
    graph.nodes.push_back({id, kind, std::move(estimate)});
    return id;
  };
  auto addEdge = [&](uint32_t from, uint32_t to, CVSplitEdgeKind kind,
                     uint32_t distance = 0) {
    graph.edges.push_back({from, to, kind, distance});
  };

  for (unsigned lane = 0; lane < lanes; ++lane) {
    std::optional<uint32_t> previousTail;
    unsigned cubeOrdinal = 0;
    for (unsigned phase = 0; phase < lineageCount; ++phase) {
      unsigned transferIndex = lane * lineageCount + phase;
      unsigned setIndex = 2 * transferIndex;
      PhaseNodes &nodes = phaseNodes[phase][lane];
      nodes.cubeToVector = requests.transferRequests[transferIndex].kind ==
                           CVSplitTransferKind::FixpipeDrain;
      if (nodes.cubeToVector) {
        unsigned cubeIndex = cubeOrdinal++ * lanes + lane;
        auto cube = model.estimateCube(requests.cubeRequests[cubeIndex]);
        if (cube.status != CVSplitPrimitiveStatus::Success)
          return failure();
        nodes.cube = addNode(CVSplitPrimitiveKind::Cube, std::move(cube));
        nodes.transfer = addNode(
            CVSplitPrimitiveKind::Transfer,
            model.estimateTransfer(requests.transferRequests[transferIndex]));
        uint32_t set = addNode(CVSplitPrimitiveKind::Synchronization,
                               model.estimateSynchronization(
                                   requests.synchronizationRequests[setIndex]));
        uint32_t wait =
            addNode(CVSplitPrimitiveKind::Synchronization,
                    model.estimateSynchronization(
                        requests.synchronizationRequests[setIndex + 1]));
        nodes.vector =
            addNode(CVSplitPrimitiveKind::VectorRegion, *vectorPhase);
        nodes.first = nodes.cube;
        nodes.tail = nodes.vector;
        addEdge(nodes.cube, nodes.transfer, CVSplitEdgeKind::DataDependency);
        addEdge(nodes.transfer, set, CVSplitEdgeKind::DataDependency);
        addEdge(set, wait, CVSplitEdgeKind::EventGenerateWait);
        addEdge(wait, nodes.vector, CVSplitEdgeKind::CrossCoreAvailability);
      } else {
        nodes.transfer = addNode(
            CVSplitPrimitiveKind::Transfer,
            model.estimateTransfer(requests.transferRequests[transferIndex]));
        uint32_t set = addNode(CVSplitPrimitiveKind::Synchronization,
                               model.estimateSynchronization(
                                   requests.synchronizationRequests[setIndex]));
        uint32_t wait =
            addNode(CVSplitPrimitiveKind::Synchronization,
                    model.estimateSynchronization(
                        requests.synchronizationRequests[setIndex + 1]));
        nodes.first = nodes.transfer;
        nodes.tail = wait;
        addEdge(nodes.transfer, set, CVSplitEdgeKind::DataDependency);
        addEdge(set, wait, CVSplitEdgeKind::EventGenerateWait);
      }
      if (previousTail)
        addEdge(*previousTail, nodes.first, CVSplitEdgeKind::DataDependency);
      previousTail = nodes.tail;
    }
  }

  llvm::DenseSet<unsigned> seenOrdinals;
  for (const CVSplitExtractedLineageSummary &lineage :
       candidate.matrixLineages) {
    if (lineage.phaseOrdinal >= c2vPhases.size() ||
        lineage.inFlightLimit == 0 || lineage.inFlightLimit > lanes ||
        !seenOrdinals.insert(lineage.phaseOrdinal).second)
      return failure();
    unsigned cubeBase = lineage.phaseOrdinal * lanes;
    uint64_t resultBytes = requests.cubeRequests[cubeBase].resultBytes;
    if (resultBytes == 0)
      return failure();
    for (unsigned lane = 1; lane < lanes; ++lane)
      if (requests.cubeRequests[cubeBase + lane].resultBytes != resultBytes)
        return failure();
    graph.matrixLineageInFlightLimits[lineage.phaseOrdinal] =
        lineage.inFlightLimit;
    graph.matrixLineageResultBytes[lineage.phaseOrdinal] = resultBytes;
    unsigned phase = c2vPhases[lineage.phaseOrdinal];
    for (unsigned lane = 0; lane < lanes; ++lane) {
      unsigned target = lane + lineage.inFlightLimit;
      uint32_t distance = target >= lanes ? 1 : 0;
      target %= lanes;
      addEdge(phaseNodes[phase][lane].transfer, phaseNodes[phase][target].cube,
              CVSplitEdgeKind::ResourceSerialization, distance);
    }
  }
  if (seenOrdinals.size() != c2vPhases.size())
    return failure();

  unsigned firstC2V = c2vPhases.front();
  unsigned lastC2V = c2vPhases.back();
  for (unsigned lane = 0; lane < lanes; ++lane) {
    unsigned nextLane = (lane + 1) % lanes;
    uint32_t distance = lane + 1 == lanes ? 1 : 0;
    addEdge(phaseNodes[lastC2V][lane].vector,
            phaseNodes[firstC2V][nextLane].vector,
            distance ? CVSplitEdgeKind::LoopCarried
                     : CVSplitEdgeKind::DataDependency,
            distance);
    addEdge(phaseNodes[lastC2V][lane].vector, phaseNodes[firstC2V][lane].cube,
            CVSplitEdgeKind::StorageReuse, 1);
  }
  return graph;
}
} // namespace

CVSplitScheduleEstimateRequest CVSplitOwnedScheduleRequest::getRequest() const {
  return {candidateId, nodes, edges, logicalUnrollFactor, modeledIterations};
}

FailureOr<llvm::SmallVector<CVSplitOwnedScheduleRequest, 0>>
buildCostModelCandidateGraphs(const CVSplitCostModelRequestSet &requests,
                              const CVSplitPrimitiveCostModel &model) {
  llvm::SmallVector<CVSplitOwnedScheduleRequest, 0> graphs;
  for (const CVSplitExtractedCandidateSummary &candidate :
       requests.candidates) {
    auto graph = buildOneGraph(requests, model, candidate);
    if (failed(graph))
      return failure();
    graphs.push_back(std::move(*graph));
  }
  return graphs;
}
} // namespace mlir::triton::cv_split
