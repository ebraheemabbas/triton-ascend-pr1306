#!/usr/bin/env python3
"""Source and reference-policy contracts for Stage 9.2 planning."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PYBIND = ROOT / "third_party/ascend/triton_ascend.cc"


def read(path: Path) -> str:
    return path.read_text()


def reference_plan(lanes: int, score_width: int, score_bytes: int,
                   probability_bytes: int, product_bytes: int) -> dict:
    depth = min(2, lanes)
    return {
        "score_depth": depth,
        "product_depth": depth,
        "probability_slots": lanes,
        "chunks": score_width // 64,
        "ub_bytes": depth * (score_bytes + product_bytes),
        "l1_bytes": lanes * probability_bytes,
        "events": 3 * lanes + 2 * depth,
        "reductions": lanes - 1,
    }


def test_option_is_default_off_and_fully_propagated() -> None:
    passes = read(INCLUDE / "Passes.td")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    backend = read(BACKEND)
    pybind = read(PYBIND)
    assert 'Option<"enableStage9SchedulePlanDiagnostics"' in passes
    option = passes.split(
        'Option<"enableStage9SchedulePlanDiagnostics"', 1)[1]
    assert '"bool", /*default*/"false"' in option.split('>,', 1)[0]
    assert "options.enableStage9SchedulePlanDiagnostics" in cpp
    assert "opts.enableStage9SchedulePlanDiagnostics" in pybind
    assert "cv_split_enable_stage9_schedule_plan_diagnostics: bool = False" in backend
    assert '"cv_split_enable_stage9_schedule_plan_diagnostics"' in backend


def test_plan_is_parameterized_verified_and_analysis_only() -> None:
    header = read(INCLUDE / "PostCVSplitSchedulePlan.h")
    source = read(LIB / "PostCVSplitSchedulePlan.cpp")
    for token in (
            "AttentionRecurrenceDescriptor",
            "PostCVSplitSchedulePlan",
            "PostCVSplitSlotAssignment",
            "PostCVSplitEventPlan",
            "PostCVSplitVectorLanePlan",
            "PostCVSplitReductionStep",
            "PostCVSplitBackendRequirements",
            "buildPostCVSplitSchedulePlan",
    ):
        assert token in header
    for token in (
            "candidate.logicalLaneCount",
            "expectedKinds",
            "operationCounts",
            "kStage9VectorChunkElements",
            "lane % plan.scoreLiveDepth",
            "lane % plan.productLiveDepth",
            "checkedAdd",
            "checkedMul",
            "verifyPlan",
            "activeValues.size() > 1",
            "pathStartsInUb",
            "cubeLineages.size() != 2",
            "PostCVSplitSchedulePlanStatus::FlagOverflow",
            "PostCVSplitSchedulePlanStatus::MemoryBudgetExceeded",
    ):
        assert token in source
    recurrence_gate = source.split("if (!reductionGeometryFound", 1)[1]
    recurrence_gate = recurrence_gate.split(") {", 1)[0]
    assert "!plan.recurrence.hasPermute" not in recurrence_gate
    assert "probabilityTransfer.destinationLayout != CVSplitLayout::NZ" in source
    assert "for (const CVSplitCubeRequest &request : requests.cubeRequests)" in source
    assert "lineage * lanes" not in source
    lowered = source.lower()
    for forbidden in (
            "_attn_fwd",
            "flash_attention",
            "native_0316",
            "head_dim ==",
            "logicalunrollfactor == 4",
            "candidateid ==",
            "setattr(",
            "replacealluseswith",
            "builder.create",
            "erase()",
    ):
        assert forbidden not in lowered


def test_reference_parameterization_and_golden_fixture() -> None:
    u2 = reference_plan(2, 128, 32768, 32768, 32768)
    u4_hd64 = reference_plan(4, 128, 32768, 32768, 16384)
    u4_hd128 = reference_plan(4, 128, 32768, 32768, 32768)
    u8 = reference_plan(8, 128, 32768, 32768, 32768)
    assert u2["events"] == 10
    assert u4_hd64["ub_bytes"] == 98304
    assert u4_hd128 == {
        "score_depth": 2,
        "product_depth": 2,
        "probability_slots": 4,
        "chunks": 2,
        "ub_bytes": 131072,
        "l1_bytes": 131072,
        "events": 16,
        "reductions": 3,
    }
    assert u8["events"] == 28
    assert u4_hd128["events"] <= 16 < u8["events"]


def test_integration_is_after_materialization_before_transfer_mutation() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    extract = cpp.index("extractCostModelRequests(")
    stage9 = cpp.index("buildPostCVSplitSchedulePlan(")
    transfer = cpp.index("insertCrossScopeTransfers(", stage9)
    assert extract < stage9 < transfer
    assert "if (enableStage9SchedulePlanDiagnostics)" in cpp
    assert "mutation=no" in read(LIB / "PostCVSplitSchedulePlan.cpp")
    before_atomic = cpp.split("if (enableStage94AtomicRewrite) {", 1)[0]
    assert "kPreserveExplicitScheduleAttr" not in before_atomic
    assert "PostCVSplitSchedulePlan.cpp" in read(LIB / "CMakeLists.txt")


if __name__ == "__main__":
    test_option_is_default_off_and_fully_propagated()
    test_plan_is_parameterized_verified_and_analysis_only()
    test_reference_parameterization_and_golden_fixture()
    test_integration_is_after_materialization_before_transfer_mutation()
    print("Stage 9.2 post-CVSplit schedule plan source contract: PASS")
