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


if __name__ == "__main__":
    test_forced_option_is_default_off_and_atomic()
    test_materializer_outlines_all_probability_regions()
    test_no_textual_or_shape_identity_policy()
    print("Stage 9.4b/c atomic materialization source contract: PASS")
