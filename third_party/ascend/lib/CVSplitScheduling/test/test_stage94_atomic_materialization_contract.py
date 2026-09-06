#!/usr/bin/env python3
"""Source contracts for Stage 9.4b/c materialization and publication."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"


def read(path: Path) -> str:
    return path.read_text()


def test_forced_option_is_default_off_and_atomic() -> None:
    passes = read(INCLUDE / "Passes.td")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    backend = read(BACKEND)
    assert 'Option<"enableStage94AtomicRewrite"' in passes
    option = passes.split('Option<"enableStage94AtomicRewrite"', 1)[1]
    assert '"bool", /*default*/"false"' in option.split('>,', 1)[0]
    assert "options.enableStage94AtomicRewrite" in cpp
    assert "cv_split_enable_stage94_atomic_rewrite: bool = False" in backend
    assert "stage94BindingReady" in cpp
    assert "stage94StructuralCandidateReady" in cpp
    assert "stage94DetachedSchedule" in cpp
    assert "kPreserveExplicitScheduleAttr" in cpp
    assert "cube=yes vector=yes" in cpp
    assert "OwningOpRef<ModuleOp> transformedModule = moduleOp.clone()" in cpp
    assert "commitModuleClone(moduleOp, *transformedModule)" in cpp
    assert "enable_stage94_cube_only" not in backend
    assert "enable_stage94_vector_only" not in backend


def test_materializer_outlines_all_probability_regions() -> None:
    header = read(INCLUDE / "ScopeSeparation.h")
    source = read(LIB / "ScopeSeparation.cpp")
    assert "materializeStage94SimdRegions" in header
    for token in (
            "outlineStage94VectorRegion",
            "outlineStage94VectorRegions",
            "VectorToCubePack &pack",
            "vector_mode",
            'StringAttr::get(context, "simd")',
            'BoolAttr::get(context, true)',
            "simdScope.setNoInline(true)",
            "outputs.insert(pack.pSrc)",
            "pack.pSrc = result->packedProbability",
            "publication=detached",
    ):
        assert token in source
    outline = source.split("outlineStage94VectorRegion", 1)[1]
    assert "scope::ScopeOp" in outline
    assert "scope::ReturnOp" in outline
    assert "SyncBlockWaitOp" in outline
    assert "SyncBlockSetOp" in outline


def test_no_textual_or_shape_identity_policy() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    scope = read(LIB / "ScopeSeparation.cpp")
    combined = (cpp + scope).lower()
    for forbidden in (
            "_attn_fwd",
            "flash_attention",
            "native_0316",
            "head_dim == 128",
            "logicalunrollfactor == 4",
            "scoreoriginid == 15",
            "productoriginid == 34",
    ):
        assert forbidden not in combined


def test_generated_row_loops_handle_empty_scf_bodies() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    assert "if (!maxBody->empty())\n    maxBody->back().erase();" in online
    assert "if (!expBody->empty())\n    expBody->back().erase();" in online
    assert "b.create<arith::ConstantIntOp>(loc, 0, 32)" in online
    assert "b.create<arith::ConstantIntOp>(loc, rows, 32)" in online
    assert online.count("create<arith::IndexCastOp>") >= 2


def test_row_reduction_identities_are_materialized_inside_simd_scope() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    assert "createReductionInit" in online
    assert "getDefiningOp<linalg::FillOp>()" in online
    assert "createReductionInit(mb, maxReduce, rowScalarType)" in online
    assert "createReductionInit(eb, sumReduce, rowScalarType)" in online


def test_loop_storage_follows_escape_lifetimes() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    scope = online.index("builder.create<scope::ScopeOp>")
    for token in ("sumRowsInit", "scaledRowsInit", "packedRowsInit"):
        assert online.index(token) < scope
    sum_storage = 'createLoopStorage(builder, maximumType, "stage94.sum-rows")'
    max_storage = 'createLoopStorage(b, maximumType, "stage94.max-rows")'
    assert sum_storage in online
    assert max_storage in online
    assert online.index(sum_storage) < scope < online.index(max_storage)
    for token in (
            "createLoopStorage",
            "tensor::EmptyOp",
            '"stage94.max-rows"',
            '"stage94.sum-rows"',
            '"stage94.scaled-rows"',
            '"stage94.packed-rows"',
    ):
        assert token in online


def test_online_softmax_marks_the_inter_loop_vector_dependency() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    marker = 'syncMark->setAttr("SYNC_IN_VF", StringAttr::get(context, "VST_VLD"))'
    assert marker in online
    assert online.index("Value maximum =") < online.index(marker)
    assert online.index(marker) < online.index("auto expLoop =")


def test_final_maximum_preserves_original_tensor_semantics() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    assert "b.create<arith::MaximumFOp>(loc, oldMaximum, maxLoop.getResult(0))" in online
    assert '"stage94.maximum"' not in online


def test_direct_nz_pack_reshapes_f32_before_truncation() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    reshape = "Value packedFloatChunk = eb.create<tensor::ReshapeOp>"
    truncate = "Value packedChunk = eb.create<arith::TruncFOp>"
    assert reshape in online
    assert truncate in online
    assert online.index(reshape) < online.index(truncate)
    assert "packedFloatChunkType, exponential, shape" in online
    assert "packedChunkType, packedFloatChunk" in online


def test_lane_scope_defers_only_the_last_logical_lane_sum() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    assert "struct Stage94OnlineSoftmaxLane" in source
    for token in (
            "bool deferLaneSum",
            "deferLaneSum ? maxLoop.getResult(1) : sumRowsInit",
            "if (deferLaneSum)",
            "simdScope->getResult(deferLaneSum ? 1 : 0)",
            "deferredBuilder.setInsertionPointAfter(anchor)",
            "deferredScope",
            "deferredLoop",
            'deferLaneSum ? "deferred" : "inline"',
    ):
        assert token in online
    assert "ValueRange{expLoop.getResult(0), maximum, expLoop.getResult(1)}" in online
    assert "ValueRange{maximum, expLoop.getResult(0), expLoop.getResult(1)}" in online
    scope = online.index("builder.create<scope::ScopeOp>")
    assert scope < online.index("simdScope.setNoInline(true)")
    assert online.index("simdScope.setNoInline(true)") < online.index(
        'simdScope->setAttr("outline"')
    worklist = online.split("SmallVector<Operation *> worklist", 1)[1].split("};", 1)[0]
    assert "alpha.getOperation()" not in worklist
    assert "newDenominator.getOperation()" not in worklist
    replacements = online.split("SmallVector<std::pair<Value, Value>> replacements", 1)[1]
    replacements = replacements.split("};", 1)[0]
    assert "newMaximum.getResult()" in replacements
    assert "sumReduce.getResult(0)" in replacements
    assert "newDenominator.getResult()" not in replacements
    assert "alpha.getResult()" not in replacements
    outline = source.split("outlineStage94VectorRegions", 1)[1]
    assert "lane + 1 == packs.size()" in outline


def test_grouped_recurrence_is_lane_count_driven_and_balanced() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    grouped = source.split("materializeStage94GroupedRecurrence", 1)[1]
    grouped = grouped.split("outlineStage94VectorRegions", 1)[0]
    for token in (
            "alphaResultTypes(lanes.size(), rowType)",
            "for (Stage94OnlineSoftmaxLane state : lanes)",
            "previousMaximum = state.maximum",
            "while (segments.size() > 1)",
            "index + 1 == segments.size()",
            "left.scale, right.scale",
            "left.offset, right.scale",
            "scaledLeft, right.offset",
            "firstState.oldDenominator",
            "segments.front().scale",
            "segments.front().offset",
            "stage94-materialized-grouped-recurrence",
    ):
        assert token in grouped
    assert "lanes.size() == 4" not in grouped
    assert "lanes.size() != 4" not in grouped


def test_grouped_recurrence_replaces_alpha_and_final_denominator_atomically() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    grouped = source.split("materializeStage94GroupedRecurrence", 1)[1]
    grouped = grouped.split("outlineStage94VectorRegions", 1)[0]
    assert "use->set(alphaScope->getResult(lane))" in grouped
    assert "use->set(affineScope->getResult(0))" in grouped
    for token in (
            "state.newDenominator.erase()",
            "state.scaledDenominator.erase()",
            "state.alpha.erase()",
            "state.alphaDifference.erase()",
    ):
        assert token in grouped
    outline = source.split("outlineStage94VectorRegions", 1)[1]
    assert "SmallVector<Stage94OnlineSoftmaxLane> lanes" in outline
    assert "return materializeStage94GroupedRecurrence(lanes);" in outline


def test_release_protocol_is_driven_by_detached_schedule_roles_and_slots() -> None:
    header = read(INCLUDE / "ScopeSeparation.h")
    source = read(LIB / "ScopeSeparation.cpp")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    assert "const PostCVSplitDetachedSchedule *" in header
    assert "stage94DetachedSchedule" in header
    assert "stage94DetachedSchedule.emplace(detachedSchedule)" in cpp
    assert "stage94DetachedSchedule ? &*stage94DetachedSchedule : nullptr" in cpp
    protocol = source.split("buildStage94ReleaseProtocolPlan", 1)[1]
    protocol = protocol.split("retileVectorScopeForRowSplit", 1)[0]
    for token in (
            "PostCVSplitDetachedCommandKind::ScorePublish",
            "PostCVSplitDetachedCommandKind::ScoreReleaseWait",
            "PostCVSplitDetachedCommandKind::ScoreRelease",
            "PostCVSplitDetachedCommandKind::ProbabilityPublish",
            "PostCVSplitDetachedCommandKind::ProductPublish",
            "PostCVSplitDetachedCommandKind::ProductReleaseWait",
            "PostCVSplitDetachedCommandKind::ProductRelease",
            "release->slot",
            "release->logicalFlagId",
            "initial.signalingResource",
            "initial.waitingResource",
            "stage94PipeForResource",
            "legacyVectorSet->erase()",
            "legacyCubeWait->erase()",
            "scoreCubeBuilder.create<hivm::SyncBlockWaitOp>",
            "scoreVectorBuilder.create<hivm::SyncBlockSetOp>",
            "productCubeBuilder.create<hivm::SyncBlockWaitOp>",
            "productVectorBuilder.create<hivm::SyncBlockSetOp>",
            "stage94-materialized-release-protocol",
    ):
        assert token in protocol
    for forbidden in (
            "scoreReleaseFlag = 12",
            "scoreReleaseFlag = 13",
            "productReleaseFlag = 14",
            "productReleaseFlag = 15",
    ):
        assert forbidden not in protocol


if __name__ == "__main__":
    test_forced_option_is_default_off_and_atomic()
    test_materializer_outlines_all_probability_regions()
    test_no_textual_or_shape_identity_policy()
    test_generated_row_loops_handle_empty_scf_bodies()
    test_row_reduction_identities_are_materialized_inside_simd_scope()
    test_loop_storage_follows_escape_lifetimes()
    test_online_softmax_marks_the_inter_loop_vector_dependency()
    test_final_maximum_preserves_original_tensor_semantics()
    test_direct_nz_pack_reshapes_f32_before_truncation()
    test_lane_scope_defers_only_the_last_logical_lane_sum()
    test_grouped_recurrence_is_lane_count_driven_and_balanced()
    test_grouped_recurrence_replaces_alpha_and_final_denominator_atomically()
    test_release_protocol_is_driven_by_detached_schedule_roles_and_slots()
    print("Stage 9.4b/c atomic materialization source contract: PASS")
