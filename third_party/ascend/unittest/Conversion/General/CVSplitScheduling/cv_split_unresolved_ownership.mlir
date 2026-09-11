// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --check-prefix=IR
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// A matmul/exp pair has a forward CUBE-to-VECTOR transfer but no event path
// proving that the VECTOR read completes before the next loop iteration reuses
// its UB slot. Classification alone is not enough to commit this schedule.
// Keep this negative beside the positive round-trip ownership fixtures.

// DIAG: [cv-split] Classification: 4C 4V
// DIAG: [cv-split] bound-resource-plan status=unresolved-ownership
// DIAG: [cv-split] verified emission plan rejected: envelope
// DIAG: [cv-split] Candidate failed; restoring function and trying next function
// DIAG: [cv-split] No candidate transformed; keeping original IR

// IR-NOT: triton_ascend.cv_split_scheduling.applied
// IR-NOT: hivm.disable_auto_tile_and_bind_subblock
// IR-LABEL: func.func @unclosed_ownership
// IR: %[[ONE:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE]] {
// IR-NEXT: %[[RESULT:.*]] = linalg.matmul
// IR-NEXT: %{{.*}} = math.exp %[[RESULT]] : tensor<32x16xf32>
// IR-NEXT: }
// IR-NEXT: return
// IR-NOT: scope.scope
// IR-NOT: ssbuffer.core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unclosed_ownership(%lhs: tensor<32x16xf16>,
                                %rhs: tensor<16x16xf16>,
                                %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %matmul : tensor<32x16xf32>
    }
    return
  }
}
