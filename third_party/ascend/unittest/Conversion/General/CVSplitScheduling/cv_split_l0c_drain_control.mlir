// RUN: %PYTHON %S/Inputs/check_l0c_drain_control.py triton-opt %s
//
// Dense attention recurrence: checks generic and full materialized drain policy.
// Locations removed from a pre-CVSplit compiler capture; no runtime data.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c128 = arith.constant 128 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c28_i32 = arith.constant 28 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c128_i32 = arith.constant 128 : i32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %3 = tensor.empty() : tensor<128xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
    scf.for %arg13 = %arg10 to %c8192_i32 step %c28_i32  : i32 {
      %6 = arith.divsi %arg13, %c8_i32 : i32
      %7 = arith.remsi %arg13, %c8_i32 : i32
      %8 = arith.divsi %6, %c8_i32 : i32
      %9 = arith.remsi %6, %c8_i32 : i32
      %10 = arith.extsi %8 : i32 to i64
      %11 = arith.muli %10, %c1048576_i64 : i64
      %12 = arith.extsi %9 : i32 to i64
      %13 = arith.muli %12, %c131072_i64 : i64
      %14 = arith.addi %11, %13 : i64
      %15 = arith.index_cast %14 : i64 to index
      %16 = arith.muli %7, %c128_i32 : i32
      %17 = arith.maxsi %16, %c0_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.muli %18, %c128 : index
      %20 = arith.addi %19, %15 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xf16>
      memref.copy %reinterpret_cast, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
      %21 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16> to tensor<128x128xf16>
      %22:5 = scf.for %arg14 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg15 = %1, %arg16 = %5, %arg17 = %4, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32)  : i32 {
        %31 = arith.maxsi %arg18, %c0_i32 : i32
        %32 = arith.index_cast %31 : i32 to index
        %33 = arith.muli %32, %c128 : index
        %34 = arith.addi %33, %15 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %35 = arith.maxsi %arg19, %c0_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = arith.muli %36, %c128 : index
        %38 = arith.addi %37, %15 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<128x128xf16>
        memref.copy %reinterpret_cast_5, %alloc_7 : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
        %39 = bufferization.to_tensor %alloc_7 restrict writable : memref<128x128xf16> to tensor<128x128xf16>
        %40 = tensor.empty() : tensor<128x128xf16>
        %transposed = linalg.transpose ins(%39 : tensor<128x128xf16>) outs(%40 : tensor<128x128xf16>) permutation = [1, 0]
        %41 = linalg.matmul {input_precision = "ieee"} ins(%21, %transposed : tensor<128x128xf16>, tensor<128x128xf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %42 = arith.mulf %41, %2 : tensor<128x128xf32>
        %reduced = linalg.reduce ins(%42 : tensor<128x128xf32>) outs(%4 : tensor<128xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %57 = arith.maximumf %in, %init : f32
            linalg.yield %57 : f32
          }
        %43 = arith.maximumf %arg17, %reduced : tensor<128xf32>
        %broadcasted_8 = linalg.broadcast ins(%43 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]
        %44 = arith.subf %42, %broadcasted_8 : tensor<128x128xf32>
        %45 = math.exp %44 : tensor<128x128xf32>
        %46 = arith.truncf %45 : tensor<128x128xf32> to tensor<128x128xf16>
        %alloc_9 = memref.alloc() : memref<128x128xf16>
        memref.copy %reinterpret_cast_6, %alloc_9 : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
        %47 = bufferization.to_tensor %alloc_9 restrict writable : memref<128x128xf16> to tensor<128x128xf16>
        %48 = linalg.fill ins(%cst_2 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
        %reduced_10 = linalg.reduce ins(%45 : tensor<128x128xf32>) outs(%48 : tensor<128xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %57 = arith.addf %in, %init : f32
            linalg.yield %57 : f32
          }
        %49 = arith.subf %arg17, %43 : tensor<128xf32>
        %50 = math.exp %49 : tensor<128xf32>
        %51 = arith.mulf %arg16, %50 : tensor<128xf32>
        %52 = arith.addf %51, %reduced_10 : tensor<128xf32>
        %broadcasted_11 = linalg.broadcast ins(%50 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]
        %53 = arith.mulf %arg15, %broadcasted_11 : tensor<128x128xf32>
        %54 = linalg.matmul {input_precision = "ieee"} ins(%46, %47 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%53 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %55 = arith.addi %arg19, %c128_i32 : i32
        %56 = arith.addi %arg18, %c128_i32 : i32
        scf.yield %54, %52, %43, %56, %55 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32
      }
      %23 = math.log %22#1 : tensor<128xf32>
      %24 = arith.addf %22#2, %23 : tensor<128xf32>
      %broadcasted = linalg.broadcast ins(%22#1 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]
      %25 = arith.divf %22#0, %broadcasted : tensor<128x128xf32>
      %26 = arith.muli %6, %c1024_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %16 : i32 to index
      %29 = arith.addi %27, %28 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %24 in writable %reinterpret_cast_4 : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
      %30 = arith.truncf %25 : tensor<128x128xf32> to tensor<128x128xf16>
      bufferization.materialize_in_destination %30 in writable %reinterpret_cast_3 : (tensor<128x128xf16>, memref<128x128xf16, strided<[128, 1], offset: ?>>) -> ()
    }
    return
  }
}
