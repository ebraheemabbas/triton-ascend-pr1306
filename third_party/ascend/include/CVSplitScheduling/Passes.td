/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef TRITON_CV_SPLIT_SCHEDULING_PASSES
#define TRITON_CV_SPLIT_SCHEDULING_PASSES

include "mlir/Pass/PassBase.td"

def CVSplitScheduling : Pass<"cv_split_scheduling", "mlir::ModuleOp"> {
  let summary = "Unroll innermost loop and split into concurrent Cube/Vector scopes";
  let constructor = "triton::createCVSplitSchedulingPass()";
  let options = [
    Option<"compileOn91095", "compile-on-910-95",
           "bool", /*default*/"false",
           "compile on 910_95">,
    Option<"unrollFactor", "unroll-factor",
           "int", /*default*/"2",
           "loop unroll factor (2, 4, or 8)">,
    Option<"enablePlanDrivenEarlyPublish",
           "enable-plan-driven-early-publish",
           "bool", /*default*/"false",
           "publish the terminal VECTOR-to-CUBE boundary in each unroll "
           "lineage at its earliest legal plan anchor instead of waiting for "
           "independent scalar/rank-1 local state. This optional experiment "
           "keeps storage, slot reuse, and arithmetic "
           "unchanged, and a missing/invalid analysis plan keeps the generic "
           "late-publish behavior.">,
    Option<"scheduleCandidateId", "schedule-candidate-id",
           "int", /*default*/"-1",
           "developer override selecting one immutable structural "
           "schedule candidate by deterministic ID. Negative diagnoses only. "
           "the initial override accepts only the structurally all-one control and keeps "
           "the qualified generic reorder unchanged.">,
    Option<"enablePurePrerequisiteHoisting",
           "enable-pure-prerequisite-hoisting",
           "bool", /*default*/"false",
           "move pure scalar/rank-1 VECTOR prerequisites before a "
           "CUBE-to-VECTOR wait when SSA proves they do not depend on the "
           "transferred value. Off by default; this optional transform "
           "changes scheduling but not arithmetic or synchronization.">,
    Option<"purePrerequisiteHoistBudgetBytes",
           "pure-prerequisite-hoist-budget-bytes",
           "int64_t", /*default*/"0",
           "maximum additional static result bytes that prerequisite hoisting may keep "
           "live across waits. Zero disables movement; unknown sizes are "
           "rejected rather than estimated.">,
    Option<"enableCostModelDiagnostics",
           "enable-cost-model-diagnostics",
           "bool", /*default*/"false",
           "extract typed cost-model requests from materialized "
           "pipeline/resource/candidate facts and emit deterministic debug "
           "diagnostics. This option does not query the model, select a "
           "candidate, or mutate IR.">,
    Option<"postSplitScheduleMode", "post-split-schedule-mode",
           "std::string", /*default*/"\"disabled\"",
           "post-CVSplit schedule handling: 'disabled' keeps the qualified "
           "generic path, 'analyze' builds and reports the complete detached "
           "schedule without mutating IR, and 'materialize' transactionally "
           "publishes both CUBE and VECTOR scopes after every proof passes">,
    Option<"promoteFullyUnrolled", "promote-fully-unrolled",
           "bool", /*default*/"true",
           "promote the scope loops left by a candidate whose unroll consumed "
           "its whole trip count; disable to keep them for inspection">,
    Option<"pipelineDistance", "pipeline-distance",
           "int", /*default*/"0",
           "how many transfer boundaries apart to software-pipeline a "
           "boundary's consuming work from the boundary that reuses its slot. "
           "Zero follows the inter-core buffer depth, which is what the shared "
           "slots need. A distance at or above the unroll factor emits every "
           "boundary's own work first and drains the consumers after, which is "
           "right when no slot is reused and the interleaving only shortens "
           "the producing core's run-ahead.">,
    Option<"privateBufferUbBudgetBytes", "private-buffer-ub-budget-bytes",
           "int64_t", /*default*/"-1",
           "spare UB, in bytes, that cross-scope pools may spend to stop "
           "reusing a slot between lanes. Negative means the pass does not "
           "know the headroom and spends what the schedule asks for; the "
           "backend's memory planner is what actually decides whether it "
           "fits, and it reports an overflow with the exact requirement when "
           "it does not. Zero declines any spend, which keeps the rotating "
           "pools and the interleaved schedule that orders them.">,
    Option<"promotePrivateBufferPools", "promote-private-buffer-pools",
           "bool", /*default*/"false",
           "additionally give individual pools one buffer per lane instead of "
           "a depth-sized rotating set, cheapest-first against the same UB "
           "budget. Off by default: at the unroll factors in use the reuse it "
           "removes is already ordered by a flag the schedule needs anyway, so "
           "it costs buffers and flags without removing a stall. Turn it on to "
           "stop depending on that ordering being emergent.">,
    Option<"sinkScaleIntoFixpipe", "sink-scale-into-fixpipe",
           "bool", /*default*/"false",
           "when a CUBE-to-VECTOR score tile's only consumer is one multiply "
           "by a constant splat, apply that constant as the fixpipe's "
           "quant_scale (QF322F32_PRE) during the transfer and drop the "
           "VECTOR-side multiply. Off by default because it measured 10.7% "
           "SLOWER on the FA kernel at hd=64 (2022 -> 2239 us): it saves 187 "
           "us of vector time but adds 305 us of fixpipe time, and the "
           "fixpipe is the contended pipe while VECTOR has slack. Kept as an "
           "option because the trade may invert on a shape whose fixpipe is "
           "idle; measure before turning it on.">,
    Option<"l0cPipelineDistance", "l0c-pipeline-distance",
           "int", /*default*/"0",
           "how many matmuls to move each CUBE-to-VECTOR drain past, so the "
           "drain overlaps them. Zero drains each accumulator immediately, "
           "which keeps every accumulator's live range disjoint from the next "
           "one's -- the memory planner then overlays them onto a single L0C "
           "address and the write-after-read ordering that creates makes the "
           "MAC array and the fixpipe take turns. One makes consecutive live "
           "ranges overlap, so the planner must keep them apart and the two "
           "units run together; it costs one more live accumulator per pool "
           "and delays that tile reaching VECTOR by one matmul, so the first "
           "lane of each pool stays eager.">,
    Option<"regroupSoftmaxMax", "regroup-softmax-max",
           "bool", /*default*/"false",
           "give the unrolled lanes of a streaming softmax one shared maximum, "
           "taken over the whole group before any exponential. The later "
           "lanes' rescale factors become one, so the accumulator is rescaled "
           "once per group instead of once per lane and the products "
           "accumulate into each other rather than each being drained back to "
           "VECTOR. Same arithmetic, different association: the group maximum "
           "is at least every lane's, so the exponents stay <= 0. Off by "
           "default: measured 1% to 9% SLOWER depending on configuration. It "
           "removes real work -- three of four product drains, four cross-core "
           "syncs, half the flag pressure -- but VECTOR can no longer start a "
           "lane's softmax when that lane's tile lands, since nothing may "
           "exponentiate until the last row max exists, and the group's score "
           "tiles all stay live, which pushes UB from 1.5% to 19% over budget. "
           "On this kernel that overlap is worth more than the work removed. "
           "It also changes results bit-for-bit, so any use needs an accuracy "
           "run behind it.">
  ];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::math::MathDialect",
    // The pass creates ops/attrs from these BiShengIR dialects (scopes,
    // tcore_type attrs, fixpipe/copy/sync transfers). They must be declared
    // so the dialect storage uniquers are initialized even when the input IR
    // contains no such ops yet (e.g. when run standalone via triton-opt).
    "mlir::hivm::HIVMDialect",
    "mlir::scope::ScopeDialect",
    "mlir::annotation::AnnotationDialect",
    "mlir::hacc::HACCDialect"
  ];
}

#endif // TRITON_CV_SPLIT_SCHEDULING_PASSES
