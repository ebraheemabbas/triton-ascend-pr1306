/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_BUFFER_SLOT_ALLOCATION_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_BUFFER_SLOT_ALLOCATION_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

namespace mlir::triton::cv_split {

enum class BufferSlotAnnotation { None, CrossCoreReadWrite };

// The adapter owns the insertion point, type, alignment, and annotation policy.
// In particular, CUBE-local CC allocations do not inherit UB/L1 shared-buffer
// annotations or view conversions merely because they use the same allocator.
inline memref::AllocOp createBufferSlotAllocation(
    OpBuilder &builder, Location loc, MemRefType type,
    BufferSlotAnnotation annotationPolicy, IntegerAttr alignment = {}) {
  auto allocation = builder.create<memref::AllocOp>(loc, type);
  if (alignment)
    allocation->setAttr("alignment", alignment);
  if (annotationPolicy == BufferSlotAnnotation::CrossCoreReadWrite) {
    auto mark = builder.create<annotation::MarkOp>(loc, allocation.getResult());
    mark->setAttr("effects",
                  builder.getArrayAttr({builder.getStringAttr("write"),
                                        builder.getStringAttr("read")}));
  }
  return allocation;
}

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_BUFFER_SLOT_ALLOCATION_H
