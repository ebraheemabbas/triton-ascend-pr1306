#!/usr/bin/env python3
"""Source-contract checks for Stage 8.2 request extraction."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PYBIND = ROOT / "third_party/ascend/triton_ascend.cc"


def read(path: Path) -> str:
    return path.read_text()


def test_stage82_option_is_default_off_and_propagated() -> None:
    passes = read(INCLUDE / "Passes.td")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    backend = read(BACKEND)
    pybind = read(PYBIND)

    assert 'Option<"enableCostModelDiagnostics"' in passes
    option = passes.split('Option<"enableCostModelDiagnostics"', 1)[1]
    assert '"bool", /*default*/"false"' in option.split('>,', 1)[0]
    assert "options.enableCostModelDiagnostics" in cpp
    assert "opts.enableCostModelDiagnostics" in pybind
    assert "cv_split_enable_cost_model_diagnostics: bool = False" in backend
    assert 'metadata["cv_split_enable_cost_model_diagnostics"]' in backend


def test_stage82_owns_array_backing_and_exact_typed_requests() -> None:
    header = read(INCLUDE / "CostModelRequestExtraction.h")
    source = read(LIB / "CostModelRequestExtraction.cpp")

    for token in (
            "CVSplitOwnedVectorRegionRequest",
            "SmallVector<CVSplitVectorOpSummary>",
            "SmallVector<CVSplitNumericDependency>",
            "CVSplitCubeRequest",
            "CVSplitTransferRequest",
            "CVSplitSynchronizationRequest",
            "CVSplitExtractedCandidateSummary",
            "CVSplitUnsupportedVectorOperation",
            "CVSplitCostModelRequestSet",
            "getRequest() const",
    ):
        assert token in header

    for token in (
            "getStaticTensorBytes",
            "convertElementType",
            "extractCubeRequest",
            "isVectorToCubeInput",
            "extractBoundaryRequests",
            "extractVectorRegions",
            "extractCandidateSummaries",
            "CVSplitTransferKind::FixpipeDrain",
            "CVSplitTransferKind::CopyAndLayoutConversion",
            "CVSplitSyncKind::EventSet",
            "CVSplitSyncKind::EventWait",
            "CVSplitSyncKind::OwnershipRelease",
            "CVSplitSyncKind::LoopSeed",
            "PartialVectorCoverage",
            "op->getResultTypes()",
            "isa<IndexType>",
    ):
        assert token in source


def test_stage82_is_read_only_kernel_agnostic_and_model_free() -> None:
    header = read(INCLUDE / "CostModelRequestExtraction.h")
    source = read(LIB / "CostModelRequestExtraction.cpp")
    text = (header + source).lower()

    forbidden = (
        "_attn_fwd",
        "flash_attention",
        "native_0316",
        "head_dim",
        "lane3",
        "unrollfactor ==",
        "candidateid ==",
        "othercalibrated",
        "movebefore",
        "moveafter",
        "replacealluseswith",
        "builder.create",
        "setattr(",
        "erase()",
        "cvsplitprimitivecostmodel",
        "estimatecube(",
        "estimateschedule(",
    )
    for token in forbidden:
        assert token not in text


def test_stage82_runs_after_materialization_before_transfer_emission() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    bind = cpp.index("bindCrossCorePipelinePlan(")
    extract = cpp.index("extractCostModelRequests(")
    transfer = cpp.index("insertCrossScopeTransfers(", extract)
    assert bind < extract < transfer
    assert "if (enableCostModelDiagnostics)" in cpp
    assert '<< "[cv-split] cost-model-inputs unavailable; qualified "' in cpp
    assert '"scheduler/emitter remains active\\n"' in cpp


def test_stage82_build_and_diagnostics_are_owned() -> None:
    cmake = read(LIB / "CMakeLists.txt")
    source = read(LIB / "CostModelRequestExtraction.cpp")
    assert "CostModelRequestExtraction.cpp" in cmake
    for token in (
            "cost-model-inputs",
            "cost-model-cube",
            "cost-model-vector-region",
            "cost-model-transfer",
            "cost-model-sync",
            "cost-model-candidate",
            "cost-model-lineage",
            "cost-model-unsupported-vector",
    ):
        assert token in source


if __name__ == "__main__":
    test_stage82_option_is_default_off_and_propagated()
    test_stage82_owns_array_backing_and_exact_typed_requests()
    test_stage82_is_read_only_kernel_agnostic_and_model_free()
    test_stage82_runs_after_materialization_before_transfer_emission()
    test_stage82_build_and_diagnostics_are_owned()
    print("Stage 8.2 cost-model request extraction source contract: PASS")
