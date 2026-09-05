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
            "pack.pSrc = *probability",
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


def test_loop_carried_storage_is_created_before_the_simd_scope() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    scope = online.index("builder.create<scope::ScopeOp>")
    for token in (
            "maxRowsInit",
            "sumRowsInit",
            "scaledRowsInit",
            "packedRowsInit",
            "maximumInit",
    ):
        assert online.index(token) < scope
    assert 'createLoopStorage(builder, maximumType, "stage94.max-rows")' in online
    assert 'createLoopStorage(builder, maximumType, "stage94.sum-rows")' in online
    for token in (
            "createLoopStorage",
            "tensor::EmptyOp",
            '"stage94.max-rows"',
            '"stage94.sum-rows"',
            '"stage94.scaled-rows"',
            '"stage94.packed-rows"',
            '"stage94.maximum"',
    ):
        assert token in online
    loop_storage = online.split("auto createLoopStorage", 1)[1].split("};", 1)[0]
    for forbidden in (
            "memref::AllocOp",
            "memref::MemorySpaceCastOp",
            "bufferization::ToTensorOp",
            'mark->setAttr("effects"',
    ):
        assert forbidden not in loop_storage


def test_online_softmax_marks_the_inter_loop_vector_dependency() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    marker = 'syncMark->setAttr("SYNC_IN_VF", StringAttr::get(context, "VST_VLD"))'
    assert marker in online
    assert online.index("Value maximum =") < online.index(marker)
    assert online.index(marker) < online.index("auto expLoop =")


def test_final_maximum_has_an_explicit_destination() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    assert 'createLoopStorage(builder, maximumType, "stage94.maximum")' in online
    assert "b.create<linalg::MapOp>" in online
    assert "ValueRange{oldMaximum, maxLoop.getResult(0)}, maximumInit" in online
    assert "nestedBuilder.create<arith::MaximumFOp>" in online
    assert "nestedBuilder.create<linalg::YieldOp>" in online
    assert "Value maximum = maximumOp->getResult(0);" in online


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


def test_lane_scope_returns_only_maximum_sum_and_packed_probability() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeStage94OnlineSoftmaxRegion", 1)[1]
    result_types = "SmallVector<Type> scopeResults{maximumType, maximumType, packedType};"
    returned = "ValueRange{maximum, expLoop.getResult(0), expLoop.getResult(1)}"
    assert result_types in online
    assert returned in online
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


if __name__ == "__main__":
    test_forced_option_is_default_off_and_atomic()
    test_materializer_outlines_all_probability_regions()
    test_no_textual_or_shape_identity_policy()
    test_generated_row_loops_handle_empty_scf_bodies()
    test_row_reduction_identities_are_materialized_inside_simd_scope()
    test_loop_carried_storage_is_created_before_the_simd_scope()
    test_online_softmax_marks_the_inter_loop_vector_dependency()
    test_final_maximum_has_an_explicit_destination()
    test_direct_nz_pack_reshapes_f32_before_truncation()
    test_lane_scope_returns_only_maximum_sum_and_packed_probability()
    print("Stage 9.4b/c atomic materialization source contract: PASS")
