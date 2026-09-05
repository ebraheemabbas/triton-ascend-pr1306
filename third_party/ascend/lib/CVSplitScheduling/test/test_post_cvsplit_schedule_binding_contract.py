#!/usr/bin/env python3
"""Source contracts for Semantic anchor binding."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PYBIND = ROOT / "third_party/ascend/triton_ascend.cc"


def read(path: Path) -> str:
    return path.read_text()


def test_option_is_default_off_and_propagated() -> None:
    passes = read(INCLUDE / "Passes.td")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    backend = read(BACKEND)
    pybind = read(PYBIND)
    assert 'Option<"enableScheduleBindingDiagnostics"' in passes
    option = passes.split(
        'Option<"enableScheduleBindingDiagnostics"', 1)[1]
    assert '"bool", /*default*/"false"' in option.split('>,', 1)[0]
    assert "options.enableScheduleBindingDiagnostics" in cpp
    assert "opts.enableScheduleBindingDiagnostics" in pybind
    assert "cv_split_enable_schedule_binding_diagnostics: bool = False" in backend
    assert '"cv_split_enable_schedule_binding_diagnostics"' in backend


def test_binding_is_semantic_complete_and_non_mutating() -> None:
    header = read(INCLUDE / "PostCVSplitScheduleBinding.h")
    source = read(LIB / "PostCVSplitScheduleBinding.cpp")
    for token in (
            "PostCVSplitLaneAnchorBinding",
            "scoreProducer",
            "probabilityProducer",
            "productProducer",
            "scoreConsumers",
            "probabilityConsumers",
            "productConsumers",
            "bindPostCVSplitScheduleAnchors",
    ):
        assert token in header
    for token in (
            "operationDependsOnValue",
            "findLaneBoundary",
            "CrossCoreDirection::VectorToCube",
            "dependentLanes == pipelinePlan.laneCount",
            "dependentLanes == 0",
            "InvalidDependencyChain",
            "InvalidEngineOwnership",
            "publicationEligible = false",
            "mutationPerformed = false",
    ):
        assert token in source
    lowered = source.lower()
    for forbidden in (
            "_attn_fwd",
            "flash_attention",
            "native_0316",
            "originid ==",
            "logicalunrollfactor == 4",
            "head_dim ==",
            "opbuilder",
            "builder.create",
            "replacealluseswith",
            "takebody",
            "erase()",
            "kpreserveexplicitscheduleattr",
    ):
        assert forbidden not in lowered


def test_binding_runs_after_detached_plan_before_transfer_mutation() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    detached = cpp.index("buildPostCVSplitDetachedSchedule(")
    binding = cpp.index("bindPostCVSplitScheduleAnchors(")
    transfer = cpp.index("insertCrossScopeTransfers(", binding)
    assert detached < binding < transfer
    assert "schedule-binding unavailable" in cpp
    assert "mutation=no" in read(LIB / "PostCVSplitScheduleBinding.cpp")
    assert "PostCVSplitScheduleBinding.cpp" in read(LIB / "CMakeLists.txt")


if __name__ == "__main__":
    test_option_is_default_off_and_propagated()
    test_binding_is_semantic_complete_and_non_mutating()
    test_binding_runs_after_detached_plan_before_transfer_mutation()
    print("Semantic anchor binding source contract: PASS")
