/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "ascend/include/CVSplitScheduling/CostModelDiagnostics.h"
#include "ascend/include/CVSplitScheduling/CVSplitCostModel.h"
#include "ascend/include/CVSplitScheduling/CostModelCandidateGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static llvm::StringRef statusName(CVSplitPrimitiveStatus status) {
  switch (status) {
  case CVSplitPrimitiveStatus::Success:
    return "success";
  case CVSplitPrimitiveStatus::InvalidRequest:
    return "invalid-request";
  case CVSplitPrimitiveStatus::TargetUnsupported:
    return "target-unsupported";
  case CVSplitPrimitiveStatus::MissingCalibration:
    return "missing-calibration";
  case CVSplitPrimitiveStatus::OutOfCalibration:
    return "out-of-calibration";
  case CVSplitPrimitiveStatus::ArithmeticOverflow:
    return "arithmetic-overflow";
  }
  return "unknown";
}

static llvm::StringRef resourceName(PrincipalResource resource) {
  switch (resource) {
  case PrincipalResource::Matrix:
    return "matrix";
  case PrincipalResource::Fixpipe:
    return "fixpipe";
  case PrincipalResource::Vector:
    return "vector";
  case PrincipalResource::Mte1:
    return "mte1";
  case PrincipalResource::Mte2:
    return "mte2";
  case PrincipalResource::Mte3:
    return "mte3";
  case PrincipalResource::ScalarControl:
    return "scalar";
  }
  return "unknown";
}

static LogicalResult logEstimate(llvm::StringRef kind, unsigned index,
                                 const CVSplitPrimitiveEstimate &estimate) {
  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] cost-model-estimate kind=" << kind
                 << " index=" << index
                 << " status=" << statusName(estimate.status)
                 << " ready=" << estimate.resultReadyCycles
                 << " ii=" << estimate.initiationIntervalCycles
                 << " read-bytes=" << estimate.bytesRead
                 << " written-bytes=" << estimate.bytesWritten
                 << " uncertainty-bp=" << estimate.uncertaintyBasisPoints
                 << " calibration=" << estimate.calibrationId.value
                 << " occupancies=" << estimate.occupancy.size() << "\n";
    for (auto [occupancyIndex, occupancy] : llvm::enumerate(estimate.occupancy))
      llvm::dbgs() << "[cv-split] cost-model-occupancy kind=" << kind
                   << " index=" << index << " occupancy=" << occupancyIndex
                   << " resource=" << resourceName(occupancy.resource)
                   << " begin=" << occupancy.beginCycle
                   << " end=" << occupancy.endCycle << "\n";
  });
  return estimate.status == CVSplitPrimitiveStatus::Success ? success()
                                                            : failure();
}

} // namespace

LogicalResult
logPrimitiveCostEstimates(const CVSplitCostModelRequestSet &requests) {
  llvm::Expected<std::unique_ptr<CVSplitPrimitiveCostModel>> modelOrError =
      CVSplitPrimitiveCostModel::createForTarget(requests.target);
  if (!modelOrError) {
    llvm::consumeError(modelOrError.takeError());
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split] cost-model-factory status=unavailable\n");
    return failure();
  }
  std::unique_ptr<CVSplitPrimitiveCostModel> model = std::move(*modelOrError);
  CVSplitCostModelInfo info = model->getInfo();
  LLVM_DEBUG(llvm::dbgs() << "[cv-split] cost-model-info api="
                          << info.apiVersion
                          << " schema=" << info.calibrationSchemaVersion
                          << " calibration=" << info.calibrationId.value
                          << " product=" << info.target.productId
                          << " revision=" << info.target.revision
                          << " mode=experimental-primitive-only\n");

  LogicalResult result = success();
  for (auto [index, request] : llvm::enumerate(requests.cubeRequests))
    if (failed(logEstimate("cube", index, model->estimateCube(request))))
      result = failure();
  for (auto [index, owned] : llvm::enumerate(requests.vectorRegionRequests))
    if (failed(logEstimate("vector", index,
                           model->estimateVectorRegion(owned.getRequest()))))
      result = failure();
  for (auto [index, request] : llvm::enumerate(requests.transferRequests))
    if (failed(
            logEstimate("transfer", index, model->estimateTransfer(request))))
      result = failure();
  for (auto [index, request] :
       llvm::enumerate(requests.synchronizationRequests))
    if (failed(logEstimate("sync", index,
                           model->estimateSynchronization(request))))
      result = failure();

  LLVM_DEBUG(llvm::dbgs() << "[cv-split] cost-model-primitive-summary status="
                          << (succeeded(result) ? "success" : "partial")
                          << " cubes=" << requests.cubeRequests.size()
                          << " vectors=" << requests.vectorRegionRequests.size()
                          << " transfers=" << requests.transferRequests.size()
                          << " syncs="
                          << requests.synchronizationRequests.size()
                          << " schedule-queried=no selection-changed=no\n");
  return result;
}

LogicalResult
logCandidateScheduleEstimates(const CVSplitCostModelRequestSet &requests) {
  auto modelOrError =
      CVSplitPrimitiveCostModel::createForTarget(requests.target);
  if (!modelOrError) {
    llvm::consumeError(modelOrError.takeError());
    return failure();
  }
  std::unique_ptr<CVSplitPrimitiveCostModel> model = std::move(*modelOrError);
  auto graphs = buildCostModelCandidateGraphs(requests, *model);
  if (failed(graphs))
    return failure();
  LogicalResult result = success();
  for (const CVSplitOwnedScheduleRequest &graph : *graphs) {
    CVSplitScheduleEstimate estimate =
        model->estimateSchedule(graph.getRequest());
    LLVM_DEBUG({
      llvm::dbgs() << "[cv-split] cost-model-graph candidate="
                   << graph.candidateId << " nodes=" << graph.nodes.size()
                   << " edges=" << graph.edges.size()
                   << " iterations=" << graph.modeledIterations << "\n";
      llvm::dbgs() << "[cv-split] cost-model-schedule candidate="
                   << estimate.candidateId << " status="
                   << (estimate.status == CVSplitCandidateStatus::Success
                           ? "success"
                           : "failed")
                   << " prologue=" << estimate.prologueCycles
                   << " ii=" << estimate.steadyStateInitiationIntervalCycles
                   << " epilogue=" << estimate.epilogueCycles
                   << " critical=" << estimate.criticalPathCycles
                   << " exposed-wait=" << estimate.exposedWaitCycles
                   << " uncertainty-bp=" << estimate.uncertaintyBasisPoints
                   << "\n";
    });
    if (estimate.status != CVSplitCandidateStatus::Success)
      result = failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[cv-split] cost-model-schedule-summary status="
                          << (succeeded(result) ? "success" : "partial")
                          << " candidates=" << graphs->size()
                          << " selection-changed=no\n");
  return result;
}

} // namespace mlir::triton::cv_split
