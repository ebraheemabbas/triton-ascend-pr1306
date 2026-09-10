/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include "ascend/include/CVSplitScheduling/L0CBufferPlan.h"
#include "ascend/include/CVSplitScheduling/BufferSlotPlan.h"
#include "ascend/include/CVSplitScheduling/HardwareConstants.h"
#include "ascend/include/CVSplitScheduling/UnrollOrigin.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::triton::cv_split {
#define DEBUG_TYPE "cv-split-scheduling"

FailureOr<L0CBufferPlan> buildL0CBufferPlan(scf::ForOp cubeLoop,
                                         unsigned logicalLaneCount) {
  auto reject = [&](llvm::StringRef reason) -> FailureOr<L0CBufferPlan> {
    cubeLoop.emitRemark() << "explicit L0C storage not applied: " << reason
                         << "; keeping the requested backend-managed schedule";
    return failure();
  };
  if (logicalLaneCount < 2 || logicalLaneCount % 2 != 0)
    return reject("two-slot static rotation requires an even lane count >= 2");
  auto function = cubeLoop->getParentOfType<func::FuncOp>();
  if (!function)
    return reject("missing function owner");
  bool existingL0C = false;
  bool externalMatrix = false;
  auto checkType = [&](Type type) {
    if (auto memref = dyn_cast<MemRefType>(type))
      if (auto space = dyn_cast_or_null<hivm::AddressSpaceAttr>(memref.getMemorySpace()))
        existingL0C |= space.getAddressSpace() == hivm::AddressSpace::L0C;
  };
  for (BlockArgument argument : function.getArguments())
    checkType(argument.getType());
  function.walk([&](Operation *operation) {
    for (Type type : operation->getOperandTypes())
      checkType(type);
    for (Type type : operation->getResultTypes())
      checkType(type);
    if (isa<linalg::MatmulOp>(operation) && operation->getBlock() != cubeLoop.getBody())
      externalMatrix = true;
  });
  if (existingL0C || externalMatrix)
    return reject("additional L0C storage or matrix lifetime is not accounted for");

  L0CBufferPlan plan;
  plan.loop = cubeLoop;
  DominanceInfo dominance(function);
  llvm::DenseMap<int64_t, unsigned> poolByOrigin;
  for (Operation &operation : *cubeLoop.getBody()) {
    auto matrix = dyn_cast<linalg::MatmulOp>(&operation);
    if (!matrix)
      continue;
    auto origin = operation.getAttrOfType<IntegerAttr>(kUnrollOriginIdAttrName);
    auto lane = operation.getAttrOfType<IntegerAttr>(kL0CLaneIdAttrName);
    if (!origin || !lane || origin.getInt() < 0 || lane.getInt() < 0 ||
        static_cast<uint64_t>(lane.getInt()) >= logicalLaneCount ||
        operation.getNumResults() != 1)
      return reject("missing or invalid typed matrix lineage/lane binding");
    auto type = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
    if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
        !type.getElementType().isF32() || type.getDimSize(0) <= 0 ||
        type.getDimSize(1) <= 0 || type.getDimSize(0) % kNzTileSize != 0 ||
        type.getDimSize(1) % kNzTileSize != 0)
      return reject("unsupported static matrix destination or NZ alignment");
    auto dps = cast<DestinationStyleOpInterface>(&operation);
    if (dps.getNumDpsInits() != 1)
      return reject("matrix does not have one DPS destination");
    Value initial = dps.getDpsInitOperand(0)->get();
    bool zeroInitialized = matchPattern(initial, m_Zero());
    if (auto fill = initial.getDefiningOp<linalg::FillOp>()) {
      zeroInitialized = fill.getInputs().size() == 1 &&
                        matchPattern(fill.getInputs()[0], m_Zero());
      if (zeroInitialized && !plan.zeroScalar &&
          dominance.dominates(fill.getInputs()[0], cubeLoop.getOperation()))
        plan.zeroScalar = fill.getInputs()[0];
    }
    if (!zeroInitialized || initial.getType() != type)
      return reject("matrix destination is not a proven zero initializer");

    L0CBufferUse use;
    use.lane = static_cast<unsigned>(lane.getInt());
    use.slot = rotatingBufferSlot(use.lane, 2);
    use.producer = &operation;
    for (Operation *reader : operation.getResult(0).getUsers()) {
      if (!isa<hivm::FixpipeOp>(reader) || reader->getBlock() != cubeLoop.getBody() ||
          reader->getNumOperands() == 0 || reader->getOperand(0) != operation.getResult(0) ||
          !operation.isBeforeInBlock(reader))
        return reject("matrix result has an unbound, escaping or non-FIXPIPE reader");
      use.readers.push_back(reader);
    }
    if (use.readers.empty())
      return reject("matrix result has no drain");
    llvm::sort(use.readers, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    auto found = poolByOrigin.find(origin.getInt());
    if (found == poolByOrigin.end()) {
      unsigned index = plan.pools.size();
      poolByOrigin.try_emplace(origin.getInt(), index);
      L0CBufferPoolPlan pool;
      pool.originId = origin.getInt();
      pool.tensorType = type;
      pool.slotCount = 2;
      uint64_t elements;
      if (!checkedBufferMultiply(type.getDimSize(0), type.getDimSize(1), elements) ||
          !checkedBufferMultiply(elements, 4, pool.bytesPerSlot))
        return reject("L0C footprint overflow");
      plan.pools.push_back(std::move(pool));
      found = poolByOrigin.find(origin.getInt());
    }
    auto &pool = plan.pools[found->second];
    if (pool.tensorType != type)
      return reject("inconsistent destination shapes in one matrix family");
    pool.uses.push_back(std::move(use));
  }
  if (plan.pools.empty())
    return reject("no matrix families");
  for (auto &pool : plan.pools) {
    if (pool.uses.size() != logicalLaneCount)
      return reject("incomplete lane coverage");
    // The current static rotation requires the lineage's execution order to
    // agree with its logical lanes. Do not guess at reordered/conditional lanes.
    for (auto [index, use] : llvm::enumerate(pool.uses))
      if (use.lane != index)
        return reject("duplicate, missing or reordered matrix lanes");
    uint64_t bytes;
    if (!checkedBufferMultiply(pool.slotCount, pool.bytesPerSlot, bytes) ||
        !checkedBufferAdd(plan.allocatedBytes, bytes, plan.allocatedBytes))
      return reject("L0C pool accounting overflow");
    for (unsigned slot = 0; slot < pool.slotCount; ++slot) {
      llvm::SmallVector<unsigned> uses;
      for (const auto &use : pool.uses)
        if (use.slot == slot)
          uses.push_back(use.lane);
      bool ordered = true;
      forEachCyclicBufferReuse(uses, [&](unsigned previous, unsigned next, bool carried) {
        pool.reuseRequirements.push_back({previous, next, carried});
        if (!carried)
          for (Operation *reader : pool.uses[previous].readers)
            ordered &= reader->isBeforeInBlock(pool.uses[next].producer);
      });
      if (!ordered)
        return reject("reuse would precede the previous drain in the chosen schedule");
    }
  }
  if (plan.allocatedBytes > kA5L0CBytes)
    return reject("explicit pools exceed physical A5 L0C capacity");
  LLVM_DEBUG(llvm::dbgs() << "[cv-split] explicit-l0c-plan families=" << plan.pools.size()
                          << " lanes=" << logicalLaneCount << " bytes=" << plan.allocatedBytes
                          << " backend-completion-check-required=yes\n");
  return plan;
}
} // namespace mlir::triton::cv_split
