/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "CVSplitExperimentalPrimitiveCosts.h"

#include <algorithm>
#include <limits>

namespace mlir::triton::cv_split::detail {
namespace {

#include "CVSplitCalibrationA5Experimental.inc"

static bool sameTarget(CVSplitTargetIdentity lhs, CVSplitTargetIdentity rhs) {
  return lhs.archFamily == rhs.archFamily && lhs.productId == rhs.productId &&
         lhs.revision == rhs.revision;
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

static bool checkedCeilDiv(uint64_t numerator, uint64_t denominator,
                           uint64_t &result) {
  if (denominator == 0)
    return false;
  result = numerator / denominator + (numerator % denominator != 0);
  return true;
}

static CVSplitPrimitiveEstimate
makeFailure(CVSplitPrimitiveStatus status, CVSplitCalibrationId calibrationId) {
  CVSplitPrimitiveEstimate estimate{};
  estimate.status = status;
  estimate.uncertaintyBasisPoints = 10000;
  estimate.calibrationId = calibrationId;
  return estimate;
}

static bool validElementType(CVSplitElementType type) {
  switch (type) {
  case CVSplitElementType::F16:
  case CVSplitElementType::BF16:
  case CVSplitElementType::F32:
  case CVSplitElementType::I8:
  case CVSplitElementType::I32:
    return true;
  }
  return false;
}

static bool validLayout(CVSplitLayout layout) {
  return layout == CVSplitLayout::ND || layout == CVSplitLayout::NZ;
}

static bool validDrainKind(CVSplitDrainKind kind) {
  switch (kind) {
  case CVSplitDrainKind::None:
  case CVSplitDrainKind::L0CToUB:
  case CVSplitDrainKind::L0CToL1:
  case CVSplitDrainKind::L0CToGlobal:
    return true;
  }
  return false;
}

static bool validResource(PrincipalResource resource) {
  switch (resource) {
  case PrincipalResource::Matrix:
  case PrincipalResource::Fixpipe:
  case PrincipalResource::Vector:
  case PrincipalResource::Mte1:
  case PrincipalResource::Mte2:
  case PrincipalResource::Mte3:
  case PrincipalResource::ScalarControl:
    return true;
  }
  return false;
}

static bool inputPathCycles(CVSplitMemoryPath path, uint64_t bytes,
                            uint64_t &cycles) {
  uint64_t startup = 0;
  uint64_t rate = 1;
  switch (path) {
  case CVSplitMemoryPath::ResidentInL0A:
  case CVSplitMemoryPath::ResidentInL0B:
    cycles = 0;
    return true;
  case CVSplitMemoryPath::L1ToL0A:
  case CVSplitMemoryPath::L1ToL0B:
    startup = kL1InputStartupCycles;
    rate = kL1InputBytesPerCycle;
    break;
  case CVSplitMemoryPath::UBToL1ToL0A:
  case CVSplitMemoryPath::UBToL1ToL0B:
    startup = kUbInputStartupCycles;
    rate = kUbInputBytesPerCycle;
    break;
  case CVSplitMemoryPath::GlobalToL1ToL0A:
  case CVSplitMemoryPath::GlobalToL1ToL0B:
    startup = kGlobalInputStartupCycles;
    rate = kGlobalInputBytesPerCycle;
    break;
  default:
    return false;
  }
  uint64_t payload;
  return checkedCeilDiv(bytes, rate, payload) &&
         checkedAdd(startup, payload, cycles);
}

static bool pathMatchesOperand(CVSplitMemoryPath path, bool lhs) {
  if (lhs)
    return path == CVSplitMemoryPath::ResidentInL0A ||
           path == CVSplitMemoryPath::L1ToL0A ||
           path == CVSplitMemoryPath::UBToL1ToL0A ||
           path == CVSplitMemoryPath::GlobalToL1ToL0A;
  return path == CVSplitMemoryPath::ResidentInL0B ||
         path == CVSplitMemoryPath::L1ToL0B ||
         path == CVSplitMemoryPath::UBToL1ToL0B ||
         path == CVSplitMemoryPath::GlobalToL1ToL0B;
}

static bool vectorOperationCycles(const CVSplitVectorOpSummary &operation,
                                  uint64_t &cycles) {
  uint64_t rate;
  switch (operation.operationClass) {
  case CVSplitVectorOpClass::ElementwiseAdd:
  case CVSplitVectorOpClass::ElementwiseSub:
  case CVSplitVectorOpClass::ElementwiseMul:
    rate = kVectorElementwiseElementsPerCycle;
    break;
  case CVSplitVectorOpClass::Maximum:
    rate = kVectorMaximumElementsPerCycle;
    break;
  case CVSplitVectorOpClass::Exp:
    rate = kVectorExpElementsPerCycle;
    break;
  case CVSplitVectorOpClass::RowReduceMax:
  case CVSplitVectorOpClass::RowReduceSum:
    rate = kVectorReductionElementsPerCycle;
    break;
  case CVSplitVectorOpClass::Cast:
    rate = kVectorCastElementsPerCycle;
    break;
  case CVSplitVectorOpClass::Permute:
    rate = kVectorPermuteElementsPerCycle;
    break;
  case CVSplitVectorOpClass::Copy:
    rate = kVectorCopyElementsPerCycle;
    break;
  case CVSplitVectorOpClass::OtherCalibrated:
    return false;
  default:
    return false;
  }
  uint64_t oneOccurrence;
  return operation.elementCount != 0 && operation.occurrenceCount != 0 &&
         checkedCeilDiv(operation.elementCount, rate, oneOccurrence) &&
         checkedMul(oneOccurrence, operation.occurrenceCount, cycles);
}

static bool addOccupancy(CVSplitPrimitiveEstimate &estimate,
                         PrincipalResource resource, uint64_t begin,
                         uint64_t duration) {
  if (duration == 0)
    return true;
  uint64_t end;
  if (!checkedAdd(begin, duration, end))
    return false;
  estimate.occupancy.push_back({resource, begin, end});
  return true;
}

} // namespace

bool isExperimentalA5Target(CVSplitTargetIdentity target) {
  return target.archFamily == CVSplitArchFamily::A5 &&
         target.productId == kExperimentalA5ProductId &&
         target.revision == kExperimentalA5Revision;
}

CVSplitCostModelInfo getExperimentalA5ModelInfo(CVSplitTargetIdentity target) {
  return CVSplitCostModelInfo{kCVSplitCostModelApiVersion,
                              kExperimentalCalibrationSchemaVersion,
                              kExperimentalCalibrationId, target};
}

CVSplitPrimitiveEstimate
estimateExperimentalCube(CVSplitTargetIdentity modelTarget,
                         CVSplitCalibrationId calibrationId,
                         const CVSplitCubeRequest &request) {
  if (!sameTarget(modelTarget, request.target))
    return makeFailure(CVSplitPrimitiveStatus::TargetUnsupported,
                       calibrationId);
  if (request.m == 0 || request.n == 0 || request.k == 0 ||
          if (request.kind != CVSplitMatrixKind::Matmul ||
              request.transposeLhs ||
              request.transposeRhs) return makeFailure(CVSplitPrimitiveStatus::
                                                           OutOfCalibration,
                                                       calibrationId);
      request.m > kMaximumCalibratedMatrixDimension ||
      request.n > kMaximumCalibratedMatrixDimension ||
      request.k > kMaximumCalibratedMatrixDimension || request.lhsBytes == 0 ||
      request.rhsBytes == 0 || request.resultBytes == 0 ||
      request.lhsBytes > kMaximumCalibratedBytes ||
      request.rhsBytes > kMaximumCalibratedBytes ||
      request.resultBytes > kMaximumCalibratedBytes ||
      !pathMatchesOperand(request.lhsPath, true) ||
      !pathMatchesOperand(request.rhsPath, false) ||
      !validLayout(request.lhsLayout) || !validLayout(request.rhsLayout) ||
      !validLayout(request.outputLayout) || !validDrainKind(request.drainKind))
    return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);
  if ((request.lhsType != CVSplitElementType::F16 &&
       request.lhsType != CVSplitElementType::BF16) ||
      (request.rhsType != CVSplitElementType::F16 &&
       request.rhsType != CVSplitElementType::BF16) ||
      request.accumulatorType != CVSplitElementType::F32)
    return makeFailure(CVSplitPrimitiveStatus::OutOfCalibration, calibrationId);

  uint64_t lhsCycles, rhsCycles, inputCycles;
  uint64_t mn, work, matrixPayload, matrixCycles;
  uint64_t bytesRead;
  if (!inputPathCycles(request.lhsPath, request.lhsBytes, lhsCycles) ||
      !inputPathCycles(request.rhsPath, request.rhsBytes, rhsCycles) ||
      !checkedAdd(lhsCycles, rhsCycles, inputCycles) ||
      !checkedMul(request.m, request.n, mn) ||
      !checkedMul(mn, request.k, work) ||
      !checkedCeilDiv(work, kMatrixMacsPerCycle, matrixPayload) ||
      !checkedAdd(kMatrixStartupCycles, matrixPayload, matrixCycles) ||
      !checkedAdd(request.lhsBytes, request.rhsBytes, bytesRead))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);

  uint64_t drainCycles = 0;
  if (request.drainKind != CVSplitDrainKind::None) {
    uint64_t drainPayload;
    if (!checkedCeilDiv(request.resultBytes, kCubeDrainBytesPerCycle,
                        drainPayload) ||
        !checkedAdd(kCubeDrainStartupCycles, drainPayload, drainCycles))
      return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                         calibrationId);
  }

  uint64_t matrixEnd, resultReady;
  if (!checkedAdd(inputCycles, matrixCycles, matrixEnd) ||
      !checkedAdd(matrixEnd, drainCycles, resultReady))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);

  CVSplitPrimitiveEstimate estimate{};
  estimate.status = CVSplitPrimitiveStatus::Success;
  estimate.resultReadyCycles = resultReady;
  estimate.initiationIntervalCycles = matrixCycles;
  estimate.bytesRead = bytesRead;
  estimate.bytesWritten = request.resultBytes;
  estimate.uncertaintyBasisPoints = kCubeUncertaintyBasisPoints;
  estimate.calibrationId = calibrationId;
  if (!addOccupancy(estimate, PrincipalResource::Mte1, 0, inputCycles) ||
      !addOccupancy(estimate, PrincipalResource::Matrix, inputCycles,
                    matrixCycles) ||
      !addOccupancy(estimate, PrincipalResource::Fixpipe, matrixEnd,
                    drainCycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);
  return estimate;
}

CVSplitPrimitiveEstimate
estimateExperimentalVectorRegion(CVSplitTargetIdentity modelTarget,
                                 CVSplitCalibrationId calibrationId,
                                 const CVSplitVectorRegionRequest &request) {
  if (!sameTarget(modelTarget, request.target))
    return makeFailure(CVSplitPrimitiveStatus::TargetUnsupported,
                       calibrationId);
  if (request.operations.empty() || request.operations.size() > 1024 ||
      !validLayout(request.inputLayout) || !validLayout(request.outputLayout) ||
      request.externalBytesRead > kMaximumCalibratedBytes ||
      request.externalBytesWritten > kMaximumCalibratedBytes ||
      request.temporaryUbBytes > kMaximumCalibratedBytes)
    return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);

  uint64_t vectorCycles = kVectorRegionStartupCycles;
  uint32_t maximumDepth = 0;
  for (const CVSplitVectorOpSummary &operation : request.operations) {
    if (!validElementType(operation.inputType) ||
        !validElementType(operation.outputType))
      return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);
    uint64_t operationCycles;
    if (!vectorOperationCycles(operation, operationCycles))
      return makeFailure(CVSplitPrimitiveStatus::OutOfCalibration,
                         calibrationId);
    if (!checkedAdd(vectorCycles, operationCycles, vectorCycles))
      return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                         calibrationId);
    maximumDepth = std::max(maximumDepth, operation.dependencyDepth);
  }
  for (const CVSplitNumericDependency &dependency : request.dependencies) {
    if (dependency.fromOperation >= request.operations.size() ||
        dependency.toOperation >= request.operations.size() ||
        dependency.fromOperation >= dependency.toOperation ||
        dependency.iterationDistance != 0)
      return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);
  }

  uint64_t dispatchCycles, depthCycles, temporaryTraffic;
  if (!checkedMul(request.oneOutlinedRegion ? 1 : request.operations.size(),
                  kVectorFragmentDispatchCycles, dispatchCycles) ||
      !checkedMul(maximumDepth, kVectorDependencyDepthCycles, depthCycles) ||
      !checkedMul(request.temporaryUbBytes, 2, temporaryTraffic) ||
      !checkedAdd(vectorCycles, dispatchCycles, vectorCycles) ||
      !checkedAdd(vectorCycles, depthCycles, vectorCycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);

  uint64_t readTraffic, writeTraffic, readCycles, writeCycles, temporaryCycles;
  if (!checkedAdd(request.externalBytesRead, request.temporaryUbBytes,
                  readTraffic) ||
      !checkedAdd(request.externalBytesWritten, request.temporaryUbBytes,
                  writeTraffic) ||
      !checkedCeilDiv(readTraffic, kVectorUbBytesPerCycle, readCycles) ||
      !checkedCeilDiv(writeTraffic, kVectorUbBytesPerCycle, writeCycles) ||
      !checkedCeilDiv(temporaryTraffic, kVectorUbBytesPerCycle,
                      temporaryCycles) ||
      !checkedAdd(vectorCycles, temporaryCycles, vectorCycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);

  CVSplitPrimitiveEstimate estimate{};
  estimate.status = CVSplitPrimitiveStatus::Success;
  estimate.resultReadyCycles =
      std::max(vectorCycles, std::max(readCycles, writeCycles));
  estimate.initiationIntervalCycles = vectorCycles;
  estimate.bytesRead = readTraffic;
  estimate.bytesWritten = writeTraffic;
  estimate.uncertaintyBasisPoints = kVectorUncertaintyBasisPoints;
  estimate.calibrationId = calibrationId;
  if (!addOccupancy(estimate, PrincipalResource::Vector, 0, vectorCycles) ||
      !addOccupancy(estimate, PrincipalResource::Mte2, 0, readCycles) ||
      !addOccupancy(estimate, PrincipalResource::Mte3, 0, writeCycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);
  return estimate;
}

CVSplitPrimitiveEstimate
estimateExperimentalTransfer(CVSplitTargetIdentity modelTarget,
                             CVSplitCalibrationId calibrationId,
                             const CVSplitTransferRequest &request) {
  if (!sameTarget(modelTarget, request.target))
    return makeFailure(CVSplitPrimitiveStatus::TargetUnsupported,
                       calibrationId);
  if (request.bytes == 0 || request.bytes > kMaximumCalibratedBytes ||
      request.rows == 0 || request.columns == 0 ||
      !validElementType(request.elementType) ||
      !validLayout(request.sourceLayout) ||
      !validLayout(request.destinationLayout))
    return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);

  PrincipalResource resource;
  uint64_t startup;
  uint64_t rate;
  switch (request.kind) {
  case CVSplitTransferKind::FixpipeDrain:
    if (request.source != CVSplitMemorySpace::L0C ||
        request.destination != CVSplitMemorySpace::UB)
      return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);
    resource = PrincipalResource::Fixpipe;
    startup = kTransferFixpipeStartupCycles;
    rate = kTransferFixpipeBytesPerCycle;
    break;
  case CVSplitTransferKind::CopyAndLayoutConversion:
  case CVSplitTransferKind::LayoutConversion:
    if (request.source != CVSplitMemorySpace::UB ||
        request.destination != CVSplitMemorySpace::L1)
      return makeFailure(CVSplitPrimitiveStatus::OutOfCalibration,
                         calibrationId);
    resource = PrincipalResource::Mte3;
    startup = kTransferConversionStartupCycles;
    rate = kTransferMteBytesPerCycle;
    break;
  case CVSplitTransferKind::Copy:
    resource = request.source == CVSplitMemorySpace::Global
                   ? PrincipalResource::Mte2
                   : PrincipalResource::Mte3;
    startup = kTransferCopyStartupCycles;
    rate = kTransferMteBytesPerCycle;
    break;
  }

default:
  return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);
  uint64_t payload, cycles;
  if (!checkedCeilDiv(request.bytes, rate, payload) ||
      !checkedAdd(startup, payload, cycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);
  if (request.rowSplit) {
    uint64_t rowGroups;
    if (!checkedCeilDiv(request.rows, kTransferRowGroup, rowGroups) ||
        !checkedAdd(cycles, rowGroups, cycles))
      return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                         calibrationId);
  }
  if (request.scalarScaleFused &&
      !checkedAdd(cycles, kTransferFusedScaleCycles, cycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);

  CVSplitPrimitiveEstimate estimate{};
  estimate.status = CVSplitPrimitiveStatus::Success;
  estimate.resultReadyCycles = cycles;
  estimate.initiationIntervalCycles = cycles;
  estimate.bytesRead = request.bytes;
  estimate.bytesWritten = request.bytes;
  estimate.uncertaintyBasisPoints = kTransferUncertaintyBasisPoints;
  estimate.calibrationId = calibrationId;
  if (!addOccupancy(estimate, resource, 0, cycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);
  return estimate;
}

CVSplitPrimitiveEstimate estimateExperimentalSynchronization(
    CVSplitTargetIdentity modelTarget, CVSplitCalibrationId calibrationId,
    const CVSplitSynchronizationRequest &request) {
  if (!sameTarget(modelTarget, request.target))
    return makeFailure(CVSplitPrimitiveStatus::TargetUnsupported,
                       calibrationId);
  if (!validResource(request.signalingResource) ||
      !validResource(request.waitingResource))
    return makeFailure(CVSplitPrimitiveStatus::InvalidRequest, calibrationId);

  uint64_t cycles;
  switch (request.kind) {
  case CVSplitSyncKind::EventSet:
    cycles = kEventSetCycles;
    break;
  case CVSplitSyncKind::EventWait:
    cycles = kEventWaitIssueCycles;
    break;
  case CVSplitSyncKind::OwnershipRelease:
    cycles = kOwnershipReleaseCycles;
    break;
  case CVSplitSyncKind::LoopSeed:
    cycles = kLoopSeedCycles;
    break;
  }
  if ((request.crossCore &&
           default : return makeFailure(CVSplitPrimitiveStatus::InvalidRequest,
                                        calibrationId);
       !checkedAdd(cycles, kCrossCoreSyncCycles, cycles)) ||
      (request.loopCarried &&
       !checkedAdd(cycles, kLoopCarriedSyncCycles, cycles)))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);

  CVSplitPrimitiveEstimate estimate{};
  estimate.status = CVSplitPrimitiveStatus::Success;
  estimate.resultReadyCycles = cycles;
  estimate.initiationIntervalCycles = cycles;
  estimate.uncertaintyBasisPoints = kSynchronizationUncertaintyBasisPoints;
  estimate.calibrationId = calibrationId;
  if (!addOccupancy(estimate, PrincipalResource::ScalarControl, 0, cycles))
    return makeFailure(CVSplitPrimitiveStatus::ArithmeticOverflow,
                       calibrationId);
  return estimate;
}

} // namespace mlir::triton::cv_split::detail
