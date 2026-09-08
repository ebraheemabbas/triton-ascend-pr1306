/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_L0C_BUFFER_PLAN_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_L0C_BUFFER_PLAN_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::cv_split {

inline constexpr llvm::StringLiteral kL0CLaneIdAttrName = "cv_split.l0c_lane";
inline constexpr llvm::StringLiteral kExplicitL0CAppliedAttrName =
    "triton_ascend.cv_split_scheduling.explicit_l0c_applied";

struct L0CBufferUse {
  unsigned lane = 0;
  unsigned slot = 0;
  Operation *producer = nullptr;
  llvm::SmallVector<Operation *> readers;
};

struct L0CBufferReuse {
  unsigned previousLane = 0;
  unsigned nextLane = 0;
  bool loopCarried = false;
};

struct L0CBufferPoolPlan {
  int64_t originId = 0;
  RankedTensorType tensorType;
  unsigned slotCount = 0;
  uint64_t bytesPerSlot = 0;
  llvm::SmallVector<L0CBufferUse> uses;
  llvm::SmallVector<L0CBufferReuse> reuseRequirements;
};

// Handles are bound after scope cloning and fill cleanup, and must be consumed
// before loop promotion or other IR mutation. This plan does not claim that
// textual order proves completion: bufferization/backend M/FIX hazards still
// have to be lowered and qualified, including every loop-carried requirement.
struct L0CBufferPlan {
  scf::ForOp loop;
  Value zeroScalar;
  uint64_t allocatedBytes = 0;
  llvm::SmallVector<L0CBufferPoolPlan> pools;
};

/// A5-only caller. Read-only admission and binding; no partial edits on reject.
FailureOr<L0CBufferPlan> buildL0CBufferPlan(scf::ForOp cubeLoop,
                                         unsigned logicalLaneCount);
void materializeL0CBufferPlan(const L0CBufferPlan &plan);

} // namespace mlir::triton::cv_split
#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_L0C_BUFFER_PLAN_H
