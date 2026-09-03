#!/usr/bin/env python3
"""Source-contract checks for Stage 8.4a deterministic scheduling."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"


def read(path: Path) -> str:
    return path.read_text()


def test_schedule_estimator_is_owned_and_reached_by_facade() -> None:
    cmake = read(LIB / "CMakeLists.txt")
    facade = read(LIB / "CVSplitCostModel.cpp")
    assert "CVSplitScheduleEstimator.cpp" in cmake
    assert "estimateDeterministicSchedule" in facade


def test_schedule_estimator_validates_the_v4_graph() -> None:
    source = read(LIB / "CVSplitScheduleEstimator.cpp")
    required = (
        "logicalUnrollFactor == 0",
        "modeledIterations == 0",
        "kMaximumExpandedNodes",
        "try_emplace(node.nodeId",
        "validatePrimitive",
        "PrimitiveUnscoreable",
        "OutOfCalibration",
        "validEdgeKind",
        "nodeOrderById.contains",
        "scheduled != expanded.size()",
        "ArithmeticOverflow",
        "ResourceConflict",
    )
    for token in required:
        assert token in source


def test_schedule_estimator_expands_and_schedules_deterministically() -> None:
    source = read(LIB / "CVSplitScheduleEstimator.cpp")
    required = (
        "edge.iterationDistance",
        "iteration * nodesPerIteration",
        "std::priority_queue",
        "left.iteration",
        "left.requestOrder",
        "left.nodeId",
        "findResourceStart",
        "appendReservations",
        "intervalsOverlap",
        "EventGenerateWait",
        "exposedWaitCycles",
    )
    for token in required:
        assert token in source


def test_schedule_estimator_reports_all_v4_outputs() -> None:
    source = read(LIB / "CVSplitScheduleEstimator.cpp")
    for field in (
            "prologueCycles",
            "steadyStateInitiationIntervalCycles",
            "epilogueCycles",
            "criticalPathCycles",
            "exposedWaitCycles",
            "busyCycles",
            "blockedCycles",
            "idleCycles",
            "firstUseCycle",
            "lastUseCycle",
            "uncertaintyBasisPoints",
    ):
        assert field in source


def test_stage84a_is_mlir_independent_and_policy_free() -> None:
    header = read(LIB / "CVSplitScheduleEstimator.h")
    source = read(LIB / "CVSplitScheduleEstimator.cpp")
    lowered = (header + source).lower()
    forbidden = (
        "mlir/ir/",
        "operation *",
        "_attn_fwd",
        "flash_attention",
        "native_0316",
        "head_dim",
        "logicalunrollfactor == 2",
        "logicalunrollfactor == 4",
        "logicalunrollfactor == 8",
        "candidateid ==",
        "selectcandidate",
        "builder.create",
        "movebefore",
        "replacealluseswith",
        "third_party/ascend/costmodel",
    )
    for token in forbidden:
        assert token not in lowered


if __name__ == "__main__":
    test_schedule_estimator_is_owned_and_reached_by_facade()
    test_schedule_estimator_validates_the_v4_graph()
    test_schedule_estimator_expands_and_schedules_deterministically()
    test_schedule_estimator_reports_all_v4_outputs()
    test_stage84a_is_mlir_independent_and_policy_free()
    print("Stage 8.4a deterministic schedule estimator source contract: PASS")
