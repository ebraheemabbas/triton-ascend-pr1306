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

#include "ascend/include/CVSplitScheduling/CostModelRequestExtraction.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

constexpr CVSplitTargetIdentity kExperimentalA5Target{CVSplitArchFamily::A5,
                                                      9579, 0};

struct ElementFacts {
  CVSplitElementType type;
  uint64_t elements;
};

static FailureOr<uint64_t> checkedAdd(uint64_t lhs, uint64_t rhs) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return failure();
  return lhs + rhs;
}

static std::optional<CVSplitElementType> convertElementType(Type type) {
  if (isa<Float16Type>(type))
    return CVSplitElementType::F16;
  if (isa<BFloat16Type>(type))
    return CVSplitElementType::BF16;
  if (isa<Float32Type>(type))
    return CVSplitElementType::F32;
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer)
    return std::nullopt;
  if (integer.getWidth() == 8)
    return CVSplitElementType::I8;
  if (integer.getWidth() == 32)
    return CVSplitElementType::I32;
  return std::nullopt;
}

static FailureOr<uint64_t> getStaticElementCount(RankedTensorType tensor) {
  if (!tensor || !tensor.hasStaticShape())
    return failure();
  uint64_t elements = 1;
  for (int64_t dimension : tensor.getShape()) {
    if (dimension < 0 ||
        (dimension != 0 && elements > std::numeric_limits<uint64_t>::max() /
                                          static_cast<uint64_t>(dimension)))
      return failure();
    elements *= static_cast<uint64_t>(dimension);
  }
  return elements;
}

static FailureOr<uint64_t> getStaticTensorBytes(RankedTensorType tensor) {
  FailureOr<uint64_t> elements = getStaticElementCount(tensor);
  if (failed(elements))
    return failure();
  uint64_t bits = tensor.getElementTypeBitWidth();
  if (bits != 0 && *elements > std::numeric_limits<uint64_t>::max() / bits)
    return failure();
  uint64_t totalBits = *elements * bits;
  if (totalBits > std::numeric_limits<uint64_t>::max() - 7)
    return failure();
  return (totalBits + 7) / 8;
}

static FailureOr<uint32_t> narrowDimension(int64_t value) {
  if (value <= 0 ||
      static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max())
    return failure();
  return static_cast<uint32_t>(value);
}

static FailureOr<ElementFacts> getOperationElementFacts(Operation *op) {
  auto inspectTensor = [](Type type) -> FailureOr<ElementFacts> {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor)
      return failure();
    FailureOr<uint64_t> elements = getStaticElementCount(tensor);
    std::optional<CVSplitElementType> element =
        convertElementType(tensor.getElementType());
    if (failed(elements) || !element)
      return failure();
    return ElementFacts{*element, *elements};
  };

  for (Value operand : op->getOperands())
    if (succeeded(inspectTensor(operand.getType())))
      return inspectTensor(operand.getType());
  for (Value result : op->getResults())
    if (succeeded(inspectTensor(result.getType())))
      return inspectTensor(result.getType());
  for (Value operand : op->getOperands()) {
    std::optional<CVSplitElementType> element =
        convertElementType(operand.getType());
    if (element)
      return ElementFacts{*element, 1};
  }
  for (Value result : op->getResults()) {
    std::optional<CVSplitElementType> element =
        convertElementType(result.getType());
    if (element)
      return ElementFacts{*element, 1};
  }
  return failure();
}

static std::optional<CVSplitElementType> getResultElementType(Operation *op) {
  for (Value result : op->getResults()) {
    Type type = result.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    if (std::optional<CVSplitElementType> converted = convertElementType(type))
      return converted;
  }
  if (op->getNumResults() != 0)
    return std::nullopt;
  FailureOr<ElementFacts> input = getOperationElementFacts(op);
  return succeeded(input) ? std::optional<CVSplitElementType>(input->type)
                          : std::nullopt;
}

static bool isBookkeeping(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "arith.constant" || name == "tensor.empty" ||
         name == "tensor.extract" || name == "tensor.extract_slice" ||
         name == "tensor.expand_shape" || name == "tensor.collapse_shape" ||
         name == "tensor.cast";
}

static std::optional<CVSplitVectorOpClass>
classifyVectorOperation(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "arith.addf" || name == "arith.addi")
    return CVSplitVectorOpClass::ElementwiseAdd;
  if (name == "arith.subf" || name == "arith.subi")
    return CVSplitVectorOpClass::ElementwiseSub;
  if (name == "arith.mulf" || name == "arith.muli")
    return CVSplitVectorOpClass::ElementwiseMul;
  if (name == "arith.maximumf" || name == "arith.maxnumf" ||
      name == "arith.maxsi")
    return CVSplitVectorOpClass::Maximum;
  if (name == "math.exp")
    return CVSplitVectorOpClass::Exp;
  if (name == "linalg.transpose")
    return CVSplitVectorOpClass::Permute;
  if (name == "linalg.fill" || name == "linalg.broadcast" ||
      name == "memref.copy" || name == "hivm.copy")
    return CVSplitVectorOpClass::Copy;
  if (name == "arith.extf" || name == "arith.truncf" || name == "arith.extsi" ||
      name == "arith.extui" || name == "arith.trunci" ||
      name == "arith.sitofp" || name == "arith.uitofp" ||
      name == "arith.fptosi" || name == "arith.fptoui" ||
      name == "arith.index_cast" || name == "arith.bitcast")
    return CVSplitVectorOpClass::Cast;

  auto reduce = dyn_cast<linalg::ReduceOp>(op);
  if (!reduce)
    return std::nullopt;
  Block &combiner = reduce.getCombiner().front();
  auto yield = dyn_cast<linalg::YieldOp>(combiner.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  Operation *combinerOp = yield.getOperand(0).getDefiningOp();
  if (!combinerOp)
    return std::nullopt;
  llvm::StringRef combinerName = combinerOp->getName().getStringRef();
  if (combinerName == "arith.maximumf" || combinerName == "arith.maxnumf" ||
      combinerName == "arith.maxsi")
    return CVSplitVectorOpClass::RowReduceMax;
  if (combinerName == "arith.addf" || combinerName == "arith.addi")
    return CVSplitVectorOpClass::RowReduceSum;
  return std::nullopt;
}

static bool isVectorToCubeInput(Value value, Operation *consumer,
                                const CrossCorePipelinePlan &plan) {
  for (const CrossCoreBoundary &boundary : plan.boundaries) {
    if (boundary.key.direction != CrossCoreDirection::VectorToCube ||
        boundary.value != value)
      continue;
    if (llvm::is_contained(boundary.consumers, consumer))
      return true;
  }
  return false;
}

static FailureOr<CVSplitCubeRequest>
extractCubeRequest(Operation *op, const CrossCorePipelinePlan &plan) {
  auto matmul = dyn_cast<linalg::MatmulOp>(op);
  if (!matmul || op->getNumOperands() < 3)
    return failure();

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Value output = op->getOperand(2);
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outputType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outputType.hasStaticShape())
    return failure();

  FailureOr<uint32_t> m = narrowDimension(outputType.getDimSize(0));
  FailureOr<uint32_t> n = narrowDimension(outputType.getDimSize(1));
  FailureOr<uint32_t> k = narrowDimension(lhsType.getDimSize(1));
  if (failed(m) || failed(n) || failed(k) ||
      lhsType.getDimSize(0) != outputType.getDimSize(0) ||
      rhsType.getDimSize(0) != lhsType.getDimSize(1) ||
      rhsType.getDimSize(1) != outputType.getDimSize(1))
    return failure();

  std::optional<CVSplitElementType> lhsElement =
      convertElementType(lhsType.getElementType());
  std::optional<CVSplitElementType> rhsElement =
      convertElementType(rhsType.getElementType());
  std::optional<CVSplitElementType> outputElement =
      convertElementType(outputType.getElementType());
  FailureOr<uint64_t> lhsBytes = getStaticTensorBytes(lhsType);
  FailureOr<uint64_t> rhsBytes = getStaticTensorBytes(rhsType);
  FailureOr<uint64_t> outputBytes = getStaticTensorBytes(outputType);
  if (!lhsElement || !rhsElement || !outputElement || failed(lhsBytes) ||
      failed(rhsBytes) || failed(outputBytes))
    return failure();

  bool lhsFromVector = isVectorToCubeInput(lhs, op, plan);
  bool rhsFromVector = isVectorToCubeInput(rhs, op, plan);
  return CVSplitCubeRequest{
      kExperimentalA5Target,
      CVSplitMatrixKind::Matmul,
      *m,
      *n,
      *k,
      *lhsElement,
      *rhsElement,
      *outputElement,
      lhsFromVector ? CVSplitLayout::NZ : CVSplitLayout::ND,
      rhsFromVector ? CVSplitLayout::NZ : CVSplitLayout::ND,
      CVSplitLayout::NZ,
      false,
      false,
      *lhsBytes,
      *rhsBytes,
      *outputBytes,
      lhsFromVector ? CVSplitMemoryPath::UBToL1ToL0A
                    : CVSplitMemoryPath::L1ToL0A,
      rhsFromVector ? CVSplitMemoryPath::UBToL1ToL0B
                    : CVSplitMemoryPath::L1ToL0B,
      CVSplitDrainKind::None};
}

static FailureOr<std::pair<uint32_t, uint32_t>>
extractBoundaryGeometry(const CrossCoreBoundary &boundary) {
  if (boundary.logicalShape.size() != 2)
    return failure();
  int64_t rows = boundary.logicalShape[0];
  if (boundary.key.direction == CrossCoreDirection::CubeToVector) {
    if (rows % 2 != 0)
      return failure();
    rows /= 2;
  }
  FailureOr<uint32_t> narrowedRows = narrowDimension(rows);
  FailureOr<uint32_t> narrowedColumns =
      narrowDimension(boundary.logicalShape[1]);
  if (failed(narrowedRows) || failed(narrowedColumns))
    return failure();
  return std::make_pair(*narrowedRows, *narrowedColumns);
}

static std::optional<PrincipalResource>
findResource(Operation *op, const CrossCorePipelinePlan &plan) {
  for (const PipelineResourceUse &use : plan.resourceUses)
    if (use.operation == op)
      return use.resource;
  return std::nullopt;
}

static LogicalResult
extractBoundaryRequests(const CrossCorePipelinePlan &plan,
                        const CrossCoreResourcePlan &resources,
                        CVSplitCostModelRequestSet &result) {
  for (const CrossCoreBoundary &boundary : plan.boundaries) {
    std::optional<CVSplitElementType> element =
        convertElementType(boundary.elementType);
    FailureOr<std::pair<uint32_t, uint32_t>> geometry =
        extractBoundaryGeometry(boundary);
    if (!element || failed(geometry) || boundary.footprintBytes == 0)
      return failure();

    bool cubeToVector =
        boundary.key.direction == CrossCoreDirection::CubeToVector;
    result.transferRequests.push_back(CVSplitTransferRequest{
        result.target,
        cubeToVector ? CVSplitTransferKind::FixpipeDrain
                     : CVSplitTransferKind::CopyAndLayoutConversion,
        cubeToVector ? CVSplitMemorySpace::L0C : CVSplitMemorySpace::UB,
        cubeToVector ? CVSplitMemorySpace::UB : CVSplitMemorySpace::L1,
        cubeToVector ? CVSplitLayout::NZ : CVSplitLayout::ND,
        cubeToVector ? CVSplitLayout::ND : CVSplitLayout::NZ, *element,
        boundary.footprintBytes, geometry->first, geometry->second, true,
        false});
    result.synchronizationRequests.push_back(CVSplitSynchronizationRequest{
        result.target, CVSplitSyncKind::EventSet, boundary.publishResource,
        boundary.consumeResource, true, false});
    result.synchronizationRequests.push_back(CVSplitSynchronizationRequest{
        result.target, CVSplitSyncKind::EventWait, boundary.publishResource,
        boundary.consumeResource, true, false});
  }

  for (const ResourceOwnershipEdge &edge : resources.ownershipEdges) {
    if (edge.ordering != ResourceOwnershipOrdering::ExplicitReleaseRequired)
      continue;
    std::optional<PrincipalResource> signaling =
        findResource(edge.lastReader, plan);
    std::optional<PrincipalResource> waiting =
        findResource(edge.nextWriter, plan);
    if (!signaling || !waiting)
      return failure();
    result.synchronizationRequests.push_back(CVSplitSynchronizationRequest{
        result.target, CVSplitSyncKind::OwnershipRelease, *signaling, *waiting,
        true, edge.loopCarried});
    if (edge.needsSeed)
      result.synchronizationRequests.push_back(CVSplitSynchronizationRequest{
          result.target, CVSplitSyncKind::LoopSeed,
          PrincipalResource::ScalarControl, *waiting, true, true});
  }
  return success();
}

static void
extractCandidateSummaries(const CrossCoreScheduleCandidateSet &candidateSet,
                          CVSplitCostModelRequestSet &result) {
  for (const CrossCoreScheduleCandidate &candidate : candidateSet.candidates) {
    CVSplitExtractedCandidateSummary summary;
    summary.candidateId = candidate.candidateId;
    summary.logicalLaneCount = candidate.logicalLaneCount;
    summary.waveWidth = candidate.waveWidth;
    summary.maximumLiveMatrixResultsPerLineage =
        candidate.maximumLiveMatrixResultsPerLineage;
    summary.prefetchLimit = candidate.prefetchLimit;
    for (const ScheduleMatrixLineageLimit &lineage :
         candidate.matrixLineageLimits)
      summary.matrixLineages.push_back({lineage.phaseOrdinal, lineage.originId,
                                        lineage.inFlightLimit,
                                        lineage.transferSlotCount});
    result.candidates.push_back(std::move(summary));
  }
}

static LogicalResult extractVectorRegions(Block *body,
                                          const Classification &classification,
                                          const CrossCorePipelinePlan &plan,
                                          CVSplitCostModelRequestSet &result) {
  llvm::DenseMap<Operation *, unsigned> order;
  llvm::DenseSet<Operation *> vectorOperations;
  SmallVector<Operation *> orderedVectorOperations;
  unsigned nextOrder = 0;
  for (Operation &operation : *body) {
    order[&operation] = nextOrder++;
    auto classIt = classification.find(&operation);
    if (classIt == classification.end() ||
        classIt->second != EngineType::VECTOR || isa<scf::YieldOp>(operation))
      continue;
    vectorOperations.insert(&operation);
    orderedVectorOperations.push_back(&operation);
  }

  llvm::DenseSet<Operation *> visited;
  for (Operation *seed : orderedVectorOperations) {
    if (!visited.insert(seed).second)
      continue;
    SmallVector<Operation *> stack{seed};
    SmallVector<Operation *> component;
    while (!stack.empty()) {
      Operation *current = stack.pop_back_val();
      component.push_back(current);
      for (Value operand : current->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        if (definition && vectorOperations.contains(definition) &&
            visited.insert(definition).second)
          stack.push_back(definition);
      }
      for (Value value : current->getResults())
        for (Operation *user : value.getUsers())
          if (vectorOperations.contains(user) && visited.insert(user).second)
            stack.push_back(user);
    }
    llvm::sort(component, [&](Operation *lhs, Operation *rhs) {
      return order.lookup(lhs) < order.lookup(rhs);
    });

    CVSplitOwnedVectorRegionRequest region;
    region.target = result.target;
    llvm::DenseMap<Operation *, unsigned> summaryIndex;
    for (Operation *operation : component) {
      if (isBookkeeping(operation))
        continue;
      std::optional<CVSplitVectorOpClass> operationClass =
          classifyVectorOperation(operation);
      FailureOr<ElementFacts> input = getOperationElementFacts(operation);
      std::optional<CVSplitElementType> output =
          getResultElementType(operation);
      if (!operationClass || failed(input) || !output) {
        ++result.unsupportedVectorOperations;
        continue;
      }

      uint32_t dependencyDepth = 0;
      for (Value operand : operation->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        auto predecessor = summaryIndex.find(definition);
        if (predecessor == summaryIndex.end())
          continue;
        dependencyDepth = std::max(
            dependencyDepth,
            region.operations[predecessor->second].dependencyDepth + 1);
      }
      unsigned index = region.operations.size();
      summaryIndex[operation] = index;
      region.operations.push_back({*operationClass, input->elements,
                                   input->type, *output, dependencyDepth, 1});
    }

    llvm::DenseSet<uint64_t> seenDependencies;
    for (Operation *operation : component) {
      auto current = summaryIndex.find(operation);
      if (current == summaryIndex.end())
        continue;
      unsigned toIndex = current->second;
      for (Value operand : operation->getOperands()) {
        auto predecessor = summaryIndex.find(operand.getDefiningOp());
        if (predecessor == summaryIndex.end())
          continue;
        uint64_t key = (static_cast<uint64_t>(predecessor->second) << 32) |
                       static_cast<uint64_t>(toIndex);
        if (seenDependencies.insert(key).second)
          region.dependencies.push_back(
              {predecessor->second, toIndex, /*iterationDistance=*/0});
      }
    }

    if (region.operations.empty())
      continue;

    llvm::DenseSet<Operation *> componentSet;
    for (Operation *operation : component)
      componentSet.insert(operation);
    for (const CrossCoreBoundary &boundary : plan.boundaries) {
      if (boundary.key.direction == CrossCoreDirection::CubeToVector) {
        bool consumed = llvm::any_of(boundary.consumers, [&](Operation *op) {
          return componentSet.contains(op);
        });
        if (!consumed)
          continue;
        FailureOr<uint64_t> bytes =
            checkedAdd(region.externalBytesRead, boundary.footprintBytes);
        if (failed(bytes))
          return failure();
        region.externalBytesRead = *bytes;
        if (region.reductionRows == 0) {
          FailureOr<std::pair<uint32_t, uint32_t>> geometry =
              extractBoundaryGeometry(boundary);
          if (failed(geometry))
            return failure();
          region.reductionRows = geometry->first;
          region.reductionWidth = geometry->second;
        }
      } else if (componentSet.contains(boundary.producer)) {
        FailureOr<uint64_t> bytes =
            checkedAdd(region.externalBytesWritten, boundary.footprintBytes);
        if (failed(bytes))
          return failure();
        region.externalBytesWritten = *bytes;
      }
    }
    result.vectorRegionRequests.push_back(std::move(region));
  }
  return success();
}

static llvm::StringRef
extractionStatusName(CVSplitRequestExtractionStatus status) {
  return status == CVSplitRequestExtractionStatus::Complete ? "complete"
                                                            : "partial";
}

static llvm::StringRef elementTypeName(CVSplitElementType type) {
  switch (type) {
  case CVSplitElementType::F16:
    return "f16";
  case CVSplitElementType::BF16:
    return "bf16";
  case CVSplitElementType::F32:
    return "f32";
  case CVSplitElementType::I8:
    return "i8";
  case CVSplitElementType::I32:
    return "i32";
  }
  return "unknown";
}

static llvm::StringRef memoryPathName(CVSplitMemoryPath path) {
  switch (path) {
  case CVSplitMemoryPath::ResidentInL0A:
    return "resident-l0a";
  case CVSplitMemoryPath::ResidentInL0B:
    return "resident-l0b";
  case CVSplitMemoryPath::L1ToL0A:
    return "l1-l0a";
  case CVSplitMemoryPath::L1ToL0B:
    return "l1-l0b";
  case CVSplitMemoryPath::UBToL1ToL0A:
    return "ub-l1-l0a";
  case CVSplitMemoryPath::UBToL1ToL0B:
    return "ub-l1-l0b";
  case CVSplitMemoryPath::GlobalToL1ToL0A:
    return "global-l1-l0a";
  case CVSplitMemoryPath::GlobalToL1ToL0B:
    return "global-l1-l0b";
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

static llvm::StringRef syncKindName(CVSplitSyncKind kind) {
  switch (kind) {
  case CVSplitSyncKind::EventSet:
    return "event-set";
  case CVSplitSyncKind::EventWait:
    return "event-wait";
  case CVSplitSyncKind::OwnershipRelease:
    return "ownership-release";
  case CVSplitSyncKind::LoopSeed:
    return "loop-seed";
  }
  return "unknown";
}

} // namespace

CVSplitVectorRegionRequest CVSplitOwnedVectorRegionRequest::getRequest() const {
  return CVSplitVectorRegionRequest{target,
                                    operations,
                                    dependencies,
                                    externalBytesRead,
                                    externalBytesWritten,
                                    temporaryUbBytes,
                                    reductionRows,
                                    reductionWidth,
                                    inputLayout,
                                    outputLayout,
                                    oneOutlinedRegion};
}

CVSplitTargetIdentity getExperimentalA5CostModelTarget() {
  return kExperimentalA5Target;
}

FailureOr<CVSplitCostModelRequestSet>
extractCostModelRequests(Block *body, const Classification &classification,
                         const CrossCorePipelinePlan &pipelinePlan,
                         const CrossCoreResourcePlan &resourcePlan,
                         const CrossCoreScheduleCandidateSet &candidateSet) {
  if (!body || pipelinePlan.laneCount == 0 || pipelinePlan.boundaries.empty() ||
      !resourcePlan.completeLaneCoverage || !resourcePlan.anchorsComplete ||
      !resourcePlan.ownershipResolved || candidateSet.candidates.empty() ||
      candidateSet.logicalLaneCount != pipelinePlan.laneCount)
    return failure();

  CVSplitCostModelRequestSet result;
  result.target = kExperimentalA5Target;
  for (const PipelineResourceUse &use : pipelinePlan.resourceUses) {
    if (use.resource != PrincipalResource::Matrix)
      continue;
    FailureOr<CVSplitCubeRequest> cube =
        extractCubeRequest(use.operation, pipelinePlan);
    if (failed(cube))
      return failure();
    result.cubeRequests.push_back(*cube);
  }
  if (result.cubeRequests.empty() ||
      failed(extractBoundaryRequests(pipelinePlan, resourcePlan, result)) ||
      failed(extractVectorRegions(body, classification, pipelinePlan, result)))
    return failure();

  extractCandidateSummaries(candidateSet, result);
  bool hasUnknownTemporaryUb =
      llvm::any_of(result.vectorRegionRequests,
                   [](const CVSplitOwnedVectorRegionRequest &region) {
                     return !region.temporaryUbBytesKnown;
                   });
  if (result.unsupportedVectorOperations != 0 || hasUnknownTemporaryUb)
    result.status = CVSplitRequestExtractionStatus::PartialVectorCoverage;
  return result;
}

void logCostModelRequests(const CVSplitCostModelRequestSet &requests) {
  LLVM_DEBUG({
    uint64_t vectorOperations = 0;
    for (const CVSplitOwnedVectorRegionRequest &region :
         requests.vectorRegionRequests)
      vectorOperations += region.operations.size();
    llvm::dbgs() << "[cv-split] cost-model-inputs status="
                 << extractionStatusName(requests.status)
                 << " arch=A5 product=" << requests.target.productId
                 << " revision=" << requests.target.revision
                 << " cubes=" << requests.cubeRequests.size()
                 << " vector-regions=" << requests.vectorRegionRequests.size()
                 << " vector-ops=" << vectorOperations << " unsupported-vector="
                 << requests.unsupportedVectorOperations
                 << " transfers=" << requests.transferRequests.size()
                 << " syncs=" << requests.synchronizationRequests.size()
                 << " candidates=" << requests.candidates.size() << "\n";

    for (auto [index, cube] : llvm::enumerate(requests.cubeRequests))
      llvm::dbgs() << "[cv-split] cost-model-cube index=" << index
                   << " m=" << cube.m << " n=" << cube.n << " k=" << cube.k
                   << " lhs-type=" << elementTypeName(cube.lhsType)
                   << " rhs-type=" << elementTypeName(cube.rhsType)
                   << " acc-type=" << elementTypeName(cube.accumulatorType)
                   << " lhs-path=" << memoryPathName(cube.lhsPath)
                   << " rhs-path=" << memoryPathName(cube.rhsPath)
                   << " lhs-bytes=" << cube.lhsBytes
                   << " rhs-bytes=" << cube.rhsBytes
                   << " result-bytes=" << cube.resultBytes << "\n";

    for (auto [index, owned] : llvm::enumerate(requests.vectorRegionRequests)) {
      CVSplitVectorRegionRequest region = owned.getRequest();
      llvm::dbgs() << "[cv-split] cost-model-vector-region index=" << index
                   << " ops=" << region.operations.size()
                   << " deps=" << region.dependencies.size()
                   << " read-bytes=" << region.externalBytesRead
                   << " written-bytes=" << region.externalBytesWritten
                   << " temporary-ub-known="
                   << (owned.temporaryUbBytesKnown ? "yes" : "no")
                   << " rows=" << region.reductionRows
                   << " width=" << region.reductionWidth
                   << " outlined=" << (region.oneOutlinedRegion ? "yes" : "no")
                   << "\n";
    }

    for (auto [index, transfer] : llvm::enumerate(requests.transferRequests))
      llvm::dbgs() << "[cv-split] cost-model-transfer index=" << index
                   << " kind="
                   << (transfer.kind == CVSplitTransferKind::FixpipeDrain
                           ? "fixpipe-drain"
                           : "copy-layout")
                   << " bytes=" << transfer.bytes << " rows=" << transfer.rows
                   << " columns=" << transfer.columns
                   << " row-split=" << (transfer.rowSplit ? "yes" : "no")
                   << " scale-fused="
                   << (transfer.scalarScaleFused ? "yes" : "no") << "\n";

    for (auto [index, sync] : llvm::enumerate(requests.synchronizationRequests))
      llvm::dbgs() << "[cv-split] cost-model-sync index=" << index
                   << " kind=" << syncKindName(sync.kind)
                   << " signal=" << resourceName(sync.signalingResource)
                   << " wait=" << resourceName(sync.waitingResource)
                   << " loop-carried=" << (sync.loopCarried ? "yes" : "no")
                   << "\n";

    for (const CVSplitExtractedCandidateSummary &candidate :
         requests.candidates) {
      llvm::dbgs() << "[cv-split] cost-model-candidate id="
                   << candidate.candidateId
                   << " lanes=" << candidate.logicalLaneCount
                   << " wave=" << candidate.waveWidth << " live-matrix="
                   << candidate.maximumLiveMatrixResultsPerLineage
                   << " prefetch=" << candidate.prefetchLimit
                   << " lineages=" << candidate.matrixLineages.size() << "\n";
      for (const CVSplitExtractedLineageSummary &lineage :
           candidate.matrixLineages)
        llvm::dbgs() << "[cv-split] cost-model-lineage candidate="
                     << candidate.candidateId
                     << " phase=" << lineage.phaseOrdinal
                     << " origin=" << lineage.originId
                     << " in-flight=" << lineage.inFlightLimit
                     << " transfer-slots=" << lineage.transferSlotCount << "\n";
    }
  });
}

} // namespace mlir::triton::cv_split
