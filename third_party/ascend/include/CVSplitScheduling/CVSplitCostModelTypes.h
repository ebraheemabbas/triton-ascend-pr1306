/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CV_SPLIT_COST_MODEL_TYPES_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CV_SPLIT_COST_MODEL_TYPES_H

#include "ascend/include/CVSplitScheduling/CVSplitTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::triton::cv_split {

enum class CVSplitArchFamily { A5 };

struct CVSplitTargetIdentity {
  CVSplitArchFamily archFamily;
  uint32_t productId;
  uint32_t revision;
};

struct CVSplitCalibrationId {
  uint64_t value;
};

constexpr uint32_t kCVSplitCostModelApiVersion = 4;

struct CVSplitCostModelInfo {
  uint32_t apiVersion;
  uint32_t calibrationSchemaVersion;
  CVSplitCalibrationId calibrationId;
  CVSplitTargetIdentity target;
};

enum class CVSplitPrimitiveStatus {
  Success,
  InvalidRequest,
  TargetUnsupported,
  MissingCalibration,
  OutOfCalibration,
  ArithmeticOverflow
};

enum class CVSplitMatrixKind { Matmul, BatchMatmul };

enum class CVSplitElementType { F16, BF16, F32, I8, I32 };

enum class CVSplitLayout { ND, NZ };

enum class CVSplitMemorySpace { Global, L1, L0A, L0B, L0C, UB };

enum class CVSplitMemoryPath {
  ResidentInL0A,
  ResidentInL0B,
  L1ToL0A,
  L1ToL0B,
  UBToL1ToL0A,
  UBToL1ToL0B,
  GlobalToL1ToL0A,
  GlobalToL1ToL0B
};

enum class CVSplitDrainKind { None, L0CToUB, L0CToL1, L0CToGlobal };

enum class CVSplitTransferKind {
  Copy,
  LayoutConversion,
  CopyAndLayoutConversion,
  FixpipeDrain
};

enum class CVSplitSyncKind { EventSet, EventWait, OwnershipRelease, LoopSeed };

struct CVSplitNumericDependency {
  uint32_t fromOperation;
  uint32_t toOperation;
  uint32_t iterationDistance;
};

enum class CVSplitEdgeKind {
  DataDependency,
  CrossCoreAvailability,
  StorageReuse,
  ResourceSerialization,
  EventGenerateWait,
  LoopCarried,
  WaveOrTail
};

enum class CVSplitCandidateStatus {
  Success,
  InvalidRequest,
  PrimitiveUnscoreable,
  ResourceConflict,
  OutOfCalibration,
  ArithmeticOverflow
};

struct CVSplitResourceSummary {
  PrincipalResource resource;
  uint64_t busyCycles;
  uint64_t blockedCycles;
  uint64_t idleCycles;
  uint64_t firstUseCycle;
  uint64_t lastUseCycle;
};

enum class CVSplitPrimitiveKind {
  Cube,
  VectorRegion,
  Transfer,
  Synchronization
};

struct CVSplitResourceOccupancy {
  PrincipalResource resource;
  uint64_t beginCycle;
  uint64_t endCycle;
};

struct CVSplitPrimitiveEstimate {
  CVSplitPrimitiveStatus status;
  uint64_t resultReadyCycles;
  uint64_t initiationIntervalCycles;
  llvm::SmallVector<CVSplitResourceOccupancy> occupancy;
  uint64_t bytesRead;
  uint64_t bytesWritten;
  uint32_t uncertaintyBasisPoints;
  CVSplitCalibrationId calibrationId;
};

struct CVSplitCubeRequest {
  CVSplitTargetIdentity target;
  CVSplitMatrixKind kind;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  CVSplitElementType lhsType;
  CVSplitElementType rhsType;
  CVSplitElementType accumulatorType;
  CVSplitLayout lhsLayout;
  CVSplitLayout rhsLayout;
  CVSplitLayout outputLayout;
  bool transposeLhs;
  bool transposeRhs;
  uint64_t lhsBytes;
  uint64_t rhsBytes;
  uint64_t resultBytes;
  CVSplitMemoryPath lhsPath;
  CVSplitMemoryPath rhsPath;
  CVSplitDrainKind drainKind;
};

enum class CVSplitVectorOpClass {
  ElementwiseAdd,
  ElementwiseSub,
  ElementwiseMul,
  Maximum,
  Exp,
  RowReduceMax,
  RowReduceSum,
  Cast,
  Permute,
  Copy,
  OtherCalibrated
};

struct CVSplitVectorOpSummary {
  CVSplitVectorOpClass operationClass;
  uint64_t elementCount;
  CVSplitElementType inputType;
  CVSplitElementType outputType;
  uint32_t dependencyDepth;
  uint32_t occurrenceCount;
};

struct CVSplitVectorRegionRequest {
  CVSplitTargetIdentity target;
  llvm::ArrayRef<CVSplitVectorOpSummary> operations;
  llvm::ArrayRef<CVSplitNumericDependency> dependencies;
  uint64_t externalBytesRead;
  uint64_t externalBytesWritten;
  uint64_t temporaryUbBytes;
  uint32_t reductionRows;
  uint32_t reductionWidth;
  CVSplitLayout inputLayout;
  CVSplitLayout outputLayout;
  bool oneOutlinedRegion;
};

struct CVSplitTransferRequest {
  CVSplitTargetIdentity target;
  CVSplitTransferKind kind;
  CVSplitMemorySpace source;
  CVSplitMemorySpace destination;
  CVSplitLayout sourceLayout;
  CVSplitLayout destinationLayout;
  CVSplitElementType elementType;
  uint64_t bytes;
  uint32_t rows;
  uint32_t columns;
  bool rowSplit;
  bool scalarScaleFused;
};

struct CVSplitSynchronizationRequest {
  CVSplitTargetIdentity target;
  CVSplitSyncKind kind;
  PrincipalResource signalingResource;
  PrincipalResource waitingResource;
  bool crossCore;
  bool loopCarried;
};

struct CVSplitCostedNode {
  uint32_t nodeId;
  CVSplitPrimitiveKind kind;
  CVSplitPrimitiveEstimate primitive;
};

struct CVSplitCostedEdge {
  uint32_t fromNode;
  uint32_t toNode;
  CVSplitEdgeKind kind;
  uint32_t iterationDistance;
};

struct CVSplitScheduleEstimateRequest {
  uint32_t candidateId;
  llvm::ArrayRef<CVSplitCostedNode> nodes;
  llvm::ArrayRef<CVSplitCostedEdge> edges;
  uint32_t logicalUnrollFactor;
  uint32_t modeledIterations;
};

struct CVSplitScheduleEstimate {
  uint32_t candidateId;
  CVSplitCandidateStatus status;
  uint64_t prologueCycles;
  uint64_t steadyStateInitiationIntervalCycles;
  uint64_t epilogueCycles;
  uint64_t criticalPathCycles;
  uint64_t exposedWaitCycles;
  llvm::SmallVector<CVSplitResourceSummary> resources;
  uint32_t uncertaintyBasisPoints;
};

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CV_SPLIT_COST_MODEL_TYPES_H
