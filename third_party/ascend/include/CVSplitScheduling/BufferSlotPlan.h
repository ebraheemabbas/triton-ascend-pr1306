/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_BUFFER_SLOT_PLAN_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_BUFFER_SLOT_PLAN_H

#include <cassert>
#include <cstdint>
#include <limits>

namespace mlir::triton::cv_split {

// Memory-neutral arithmetic. Adapters supply their physical footprint (for
// example, ROW_SPLIT UB shards versus full matrix destinations in L0C).
inline bool checkedBufferAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

inline bool checkedBufferMultiply(uint64_t lhs, uint64_t rhs,
                                  uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

// Slot count is a storage decision, not a drain window or an event count.
inline unsigned rotatingBufferSlot(unsigned lane, unsigned slotCount) {
  assert(slotCount != 0 && "a rotating pool must have at least one slot");
  return lane % slotCount;
}

// Enumerate reuse requirements in an already validated execution order. The
// adapter must bind the previous readers/next writer and prove completion;
// enumeration alone does not establish asynchronous safety. A singleton still
// has a loop-carried reuse, while an unused slot has none.
template <typename Range, typename Visitor>
void forEachCyclicBufferReuse(const Range &orderedUses, Visitor visit) {
  if (orderedUses.empty())
    return;
  for (unsigned index = 1; index < orderedUses.size(); ++index)
    visit(orderedUses[index - 1], orderedUses[index], false);
  visit(orderedUses.back(), orderedUses.front(), true);
}

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_BUFFER_SLOT_PLAN_H
