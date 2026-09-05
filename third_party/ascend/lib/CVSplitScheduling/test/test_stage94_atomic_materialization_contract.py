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
    for token in ("maxRowsInit", "sumRowsInit", "scaledRowsInit", "packedRowsInit"):
        assert online.index(token) < scope
    assert 'createUbBackedTensor(builder, maximumType, "stage94.max-rows")' in online
    assert 'createUbBackedTensor(builder, maximumType, "stage94.sum-rows")' in online
    for token in (
            "createUbBackedTensor",
            "hivm::AddressSpace::UB",
            "memref::AllocOp",
            'mark->setAttr("effects"',
            "memref::MemorySpaceCastOp",
            "bufferization::ToTensorOp",
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


if __name__ == "__main__":
    test_forced_option_is_default_off_and_atomic()
    test_materializer_outlines_all_probability_regions()
    test_no_textual_or_shape_identity_policy()
    test_generated_row_loops_handle_empty_scf_bodies()
    test_row_reduction_identities_are_materialized_inside_simd_scope()
    test_loop_carried_storage_is_created_before_the_simd_scope()
    test_online_softmax_marks_the_inter_loop_vector_dependency()
    print("Stage 9.4b/c atomic materialization source contract: PASS")
