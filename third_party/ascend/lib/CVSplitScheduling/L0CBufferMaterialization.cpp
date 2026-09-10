/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include "ascend/include/CVSplitScheduling/L0CBufferPlan.h"
#include "ascend/include/CVSplitScheduling/BufferSlotAllocation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir::triton::cv_split {

void materializeL0CBufferPlan(const L0CBufferPlan &plan) {
  // Called only after complete read-only admission, and after fill hoisting.
  OpBuilder builder(plan.loop);
  Location loc = plan.loop.getLoc();
  Value zero = plan.zeroScalar;
  if (!zero)
    zero = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0));
  auto cc = hivm::AddressSpaceAttr::get(builder.getContext(), hivm::AddressSpace::L0C);
  for (const auto &pool : plan.pools) {
    auto bufferType = MemRefType::get(pool.tensorType.getShape(),
                                     pool.tensorType.getElementType(), nullptr, cc);
    llvm::SmallVector<Value> tensors;
    for (unsigned slot = 0; slot < pool.slotCount; ++slot) {
      auto allocation = createBufferSlotAllocation(
          builder, loc, bufferType, BufferSlotAnnotation::None,
          builder.getI64IntegerAttr(64));
      tensors.push_back(builder.create<bufferization::ToTensorOp>(
          loc, pool.tensorType, allocation.getResult(), true, true).getResult());
    }
    for (const auto &use : pool.uses) {
      OpBuilder initBuilder(use.producer);
      Value initial = initBuilder.create<linalg::FillOp>(
          use.producer->getLoc(), ValueRange{zero}, ValueRange{tensors[use.slot]}).getResult(0);
      cast<DestinationStyleOpInterface>(use.producer).getDpsInitOperand(0)->set(initial);
    }
  }
}
} // namespace mlir::triton::cv_split
