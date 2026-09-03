#!/usr/bin/env python3
"""Source-contract checks for Stage 8.3 provisional primitive costs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"


def read(path: Path) -> str:
    return path.read_text()


def test_calibration_is_isolated_versioned_and_experimental() -> None:
    calibration = read(LIB / "CVSplitCalibrationA5Experimental.inc")
    assert "kExperimentalCalibrationSchemaVersion = 4" in calibration
    assert "0xA500000000000806ULL" in calibration
    assert "kVectorParallelSubBlocks = 2" in calibration
    assert "kExperimentalA5ProductId = 9579" in calibration
    assert "not qualified absolute hardware timings" in calibration
    for category in (
            "Matrix",
            "Vector",
            "Transfer",
            "Synchronization",
            "UncertaintyBasisPoints",
    ):
        assert category in calibration


def test_every_v4_primitive_query_has_a_replaceable_body() -> None:
    facade = read(LIB / "CVSplitCostModel.cpp")
    implementation = read(LIB / "CVSplitExperimentalPrimitiveCosts.cpp")
    internal = read(LIB / "CVSplitExperimentalPrimitiveCosts.h")

    for name in (
            "estimateExperimentalCube",
            "estimateExperimentalVectorRegion",
            "estimateExperimentalTransfer",
            "estimateExperimentalSynchronization",
    ):
        assert name in internal
        assert name in implementation
        assert name in facade
    assert "CVSplitCalibrationA5Experimental.inc" in implementation
    assert "CVSplitExperimentalPrimitiveCosts.h" in facade


def test_provisional_body_is_checked_generic_and_has_statuses() -> None:
    implementation = read(LIB / "CVSplitExperimentalPrimitiveCosts.cpp")
    lowered = implementation.lower()
    for token in (
            "checkedAdd",
            "checkedMul",
            "checkedCeilDiv",
            "addOccupancy",
            "CVSplitPrimitiveStatus::Success",
            "CVSplitPrimitiveStatus::InvalidRequest",
            "CVSplitPrimitiveStatus::TargetUnsupported",
            "CVSplitPrimitiveStatus::OutOfCalibration",
            "CVSplitPrimitiveStatus::ArithmeticOverflow",
            "PrincipalResource::Matrix",
            "PrincipalResource::Fixpipe",
            "PrincipalResource::Vector",
            "PrincipalResource::Mte1",
            "PrincipalResource::Mte2",
            "PrincipalResource::Mte3",
            "PrincipalResource::ScalarControl",
    ):
        assert token in implementation

    forbidden = (
        "_attn_fwd",
        "flash_attention",
        "native_0316",
        "head_dim",
        "lane3",
        "unrollfactor ==",
        "candidateid ==",
        "third_party/ascend/costmodel",
        "ascendmodel",
    )
    for token in forbidden:
        assert token not in lowered


def test_stage83_primitive_diagnostics_remain_present() -> None:
    header = read(INCLUDE / "CostModelDiagnostics.h")
    source = read(LIB / "CostModelDiagnostics.cpp")
    pass_source = read(LIB / "CVSplitScheduling.cpp")

    assert "logPrimitiveCostEstimates" in header
    assert "createForTarget" in source
    for query in (
            "estimateCube",
            "estimateVectorRegion",
            "estimateTransfer",
            "estimateSynchronization",
    ):
        assert query in source
    assert "cost-model-primitive-summary" in source
    assert "logPrimitiveCostEstimates" in pass_source
    assert "qualified scheduler/emitter remains active" not in source
    assert "qualified scheduler/emitter remains active" not in pass_source.split("logPrimitiveCostEstimates", 1)[0]


def test_stage83_build_owns_new_sources() -> None:
    cmake = read(LIB / "CMakeLists.txt")
    assert "CVSplitExperimentalPrimitiveCosts.cpp" in cmake
    assert "CostModelDiagnostics.cpp" in cmake


if __name__ == "__main__":
    test_calibration_is_isolated_versioned_and_experimental()
    test_every_v4_primitive_query_has_a_replaceable_body()
    test_provisional_body_is_checked_generic_and_has_statuses()
    test_stage83_primitive_diagnostics_remain_present()
    test_stage83_build_owns_new_sources()
    print("Stage 8.3 provisional primitive cost source contract: PASS")
