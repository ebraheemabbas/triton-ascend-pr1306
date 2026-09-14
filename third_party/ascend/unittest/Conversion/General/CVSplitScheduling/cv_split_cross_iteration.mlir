// RUN: triton-opt %s --split-input-file "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" --debug-only=cv-split-scheduling 2>&1 | FileCheck %s --check-prefixes=CHECK,FORWARDED
// RUN: triton-opt %s --split-input-file "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" --debug-only=cv-split-scheduling 2>&1 | FileCheck %s --check-prefixes=CHECK,FORWARDED
// RUN: triton-opt %s --split-input-file "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8" --debug-only=cv-split-scheduling 2>&1 | FileCheck %s --check-prefixes=CHECK,FORWARDED
// RUN: sed 's/%limit = arith.constant 16 : index/%limit = arith.constant 2 : index/' %s | triton-opt --split-input-file "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" --debug-only=cv-split-scheduling 2>&1 | FileCheck %s --check-prefixes=CHECK,FULL2
// RUN: sed 's/%limit = arith.constant 16 : index/%limit = arith.constant 4 : index/' %s | triton-opt --split-input-file "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" --debug-only=cv-split-scheduling 2>&1 | FileCheck %s --check-prefixes=CHECK,FORWARDED
// RUN: sed 's/%limit = arith.constant 16 : index/%limit = arith.constant 8 : index/' %s | triton-opt --split-input-file "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8" --debug-only=cv-split-scheduling 2>&1 | FileCheck %s --check-prefixes=CHECK,FORWARDED
// RUN: %PYTHON %S/Inputs/check_cross_iteration.py triton-opt %s

// Isolate modules so existing flag allocations cannot mask the new precheck.
// Positive cases assert this admission gate, not that unrelated later gates
// must accept each tiny toy kernel. Full FA positive coverage is separate.
// The full-unroll RUNs ensure original carried uses cannot disappear as
// ordinary same-body edges. A single-iteration scaffold has no real backedge.

// CHECK-LABEL: [cv-split] Function: reject_c2v
// CHECK: CVSplit cross-iteration pre-check rejected: CUBE -> VECTOR
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @reject_c2v(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %zero) -> tensor<32x32xf32> {
      %use = math.exp %previous : tensor<32x32xf32>
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %score : tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: reject_v2c
// CHECK: CVSplit cross-iteration pre-check rejected: VECTOR -> CUBE
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @reject_v2c(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %lhs) -> tensor<32x32xf16> {
      %score = linalg.matmul ins(%previous, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>
      %use = math.exp %score : tensor<32x32xf32>
      %next = arith.truncf %use : tensor<32x32xf32> to tensor<32x32xf16>
      scf.yield %next : tensor<32x32xf16>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: allow_c2c
// CHECK: [cv-split] Cross-iteration consumer pre-check passed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @allow_c2c(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %zero) -> tensor<32x32xf32> {
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%previous : tensor<32x32xf32>) -> tensor<32x32xf32>
      %use = math.exp %score : tensor<32x32xf32>
      scf.yield %score : tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: allow_v2v
// CHECK: [cv-split] Cross-iteration consumer pre-check passed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @allow_v2v(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %zero) -> tensor<32x32xf32> {
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>
      %next = arith.addf %previous, %score : tensor<32x32xf32>
      scf.yield %next : tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: allow_invariant
// CHECK: [cv-split] Cross-iteration consumer pre-check passed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @allow_invariant(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %zero) -> tensor<32x32xf32> {
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%previous : tensor<32x32xf32>) -> tensor<32x32xf32>
      %use = math.exp %previous : tensor<32x32xf32>
      scf.yield %previous : tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: reject_forwarded_c2v
// FORWARDED: CVSplit cross-iteration pre-check rejected: CUBE -> VECTOR
// With only two actual iterations, this two-step forwarding chain never reads
// the produced score. Do not invent another iteration of the full-unroll scaffold.
// FULL2: [cv-split] Cross-iteration consumer pre-check passed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @reject_forwarded_c2v(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result:2 = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %zero, %forward = %zero) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
      %use = math.exp %previous : tensor<32x32xf32>
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %forward, %score : tensor<32x32xf32>, tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: allow_permuted_initials
// CHECK: [cv-split] Cross-iteration consumer pre-check passed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @allow_permuted_initials(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>, %other: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result:2 = scf.for %i = %c0 to %limit step %c1 iter_args(%a = %zero, %b = %other) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%a : tensor<32x32xf32>) -> tensor<32x32xf32>
      %use = math.exp %b : tensor<32x32xf32>
      scf.yield %b, %a : tensor<32x32xf32>, tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: allow_separated_accumulator
// CHECK: [cv-split] Cross-iteration consumer pre-check passed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @allow_separated_accumulator(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %zero) -> tensor<32x32xf32> {
      %scaled = arith.mulf %previous, %zero : tensor<32x32xf32>
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%scaled : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %score : tensor<32x32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: [cv-split] Function: reject_nested_c2v
// CHECK: CVSplit cross-iteration pre-check rejected: CUBE -> VECTOR
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @reject_nested_c2v(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>, %rowInit: tensor<32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%previous = %zero) -> tensor<32x32xf32> {
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>
      %use = linalg.reduce ins(%zero : tensor<32x32xf32>) outs(%rowInit : tensor<32xf32>) dimensions = [1]
          (%in: f32, %sum: f32) {
            %carried = tensor.extract %previous[%c0, %c0] : tensor<32x32xf32>
            %next = arith.addf %in, %carried : f32
            linalg.yield %next : f32
          }
      scf.yield %score : tensor<32x32xf32>
    }
    return
  }
}

// -----

// Loop-control arithmetic is reproduced in both scopes. A provisional VECTOR
// stamp on its last increment does not imply a cross-core data transfer.
// CHECK-LABEL: [cv-split] Function: allow_replicated_index
// CHECK: [cv-split] Cross-iteration consumer pre-check passed
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @allow_replicated_index(%source: memref<?xf16>, %rhs: tensor<32x32xf16>, %zero: tensor<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.constant 16 : index
    %result = scf.for %i = %c0 to %limit step %c1 iter_args(%offset = %c0) -> index {
      %view = memref.reinterpret_cast %source to offset: [%offset], sizes: [32, 32], strides: [32, 1]
          : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      %copy = memref.alloc() : memref<32x32xf16>
      memref.copy %view, %copy : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
      %lhs = bufferization.to_tensor %copy restrict writable : memref<32x32xf16> to tensor<32x32xf16>
      %score = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
          outs(%zero : tensor<32x32xf32>) -> tensor<32x32xf32>
      %use = math.exp %score : tensor<32x32xf32>
      %next = arith.addi %offset, %c1 : index
      scf.yield %next : index
    }
    return
  }
}
