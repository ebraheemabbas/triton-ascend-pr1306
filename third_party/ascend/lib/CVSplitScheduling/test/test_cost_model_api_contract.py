#!/usr/bin/env python3
"""Source-contract checks for the cost-model v4 API."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"


def read(path: Path) -> str:
    return path.read_text()


def test_principal_resource_has_one_mlir_independent_owner() -> None:
    shared = read(INCLUDE / "CVSplitTypes.h")
    pipeline = read(INCLUDE / "CrossCorePipelinePlan.h")
    model_types = read(INCLUDE / "CVSplitCostModelTypes.h")

    assert shared.count("enum class PrincipalResource") == 1
    assert "enum class PrincipalResource" not in pipeline
    assert 'CVSplitScheduling/CVSplitTypes.h' in pipeline
    assert 'CVSplitScheduling/CVSplitTypes.h' in model_types
    assert "mlir/" not in shared
    assert "llvm/" not in shared


def test_v4_core_types_and_queries_are_declared() -> None:
    types = read(INCLUDE / "CVSplitCostModelTypes.h")
    api = read(INCLUDE / "CVSplitCostModel.h")

    required_types = (
        "CVSplitTargetIdentity",
        "CVSplitCostModelInfo",
        "CVSplitCubeRequest",
        "CVSplitVectorRegionRequest",
        "CVSplitTransferRequest",
        "CVSplitSynchronizationRequest",
        "CVSplitPrimitiveEstimate",
        "CVSplitScheduleEstimateRequest",
        "CVSplitScheduleEstimate",
        "CVSplitResourceOccupancy",
        "CVSplitResourceSummary",
    )
    for token in required_types:
        assert f"struct {token}" in types

    assert "kCVSplitCostModelApiVersion = 4" in types
    for query in (
            "getInfo",
            "estimateCube",
            "estimateVectorRegion",
            "estimateTransfer",
            "estimateSynchronization",
            "estimateSchedule",
    ):
        assert query in api

    assert "createForTarget" in api
    assert "std::unique_ptr<Impl> impl_" in api
    assert "CVSplitPrimitiveCostModel(const CVSplitPrimitiveCostModel &) = delete" in api
    assert "CVSplitPrimitiveCostModel(CVSplitPrimitiveCostModel &&) = delete" in api


def test_facade_remains_replaceable() -> None:
    source = read(LIB / "CVSplitCostModel.cpp")

    assert "CVSplitExperimentalPrimitiveCosts.h" in source
    assert "detail::isExperimentalA5Target" in source
    assert "detail::getExperimentalA5ModelInfo" in source
    assert "createStringError" in source
    assert "~CVSplitPrimitiveCostModel() = default" in source


def test_cost_model_api_has_no_mlir_or_kernel_policy_dependency() -> None:
    files = (
        INCLUDE / "CVSplitTypes.h",
        INCLUDE / "CVSplitCostModelTypes.h",
        INCLUDE / "CVSplitCostModel.h",
        LIB / "CVSplitCostModel.cpp",
    )
    text = "\n".join(read(path) for path in files)
    lowered = text.lower()

    forbidden = (
        "mlir/ir/",
        "operation *",
        "_attn_fwd",
        "flash_attention",
        "native_0316",
        "head_dim",
        "unrollfactor ==",
        "candidateid ==",
        "third_party/ascend/costmodel",
        "ascendmodel",
    )
    for token in forbidden:
        assert token not in lowered

    pass_source = read(LIB / "CVSplitScheduling.cpp")
    assert "CVSplitPrimitiveCostModel" not in pass_source


def test_cost_model_build_owns_the_facade() -> None:
    cmake = read(LIB / "CMakeLists.txt")
    assert "CVSplitCostModel.cpp" in cmake


def test_api_surface_does_not_expose_mutation() -> None:
    api = read(INCLUDE / "CVSplitCostModel.h")
    forbidden = re.compile(r"(select|mutate|rewrite|apply)[A-Z]", re.I)
    assert not forbidden.search(api)


if __name__ == "__main__":
    test_principal_resource_has_one_mlir_independent_owner()
    test_v4_core_types_and_queries_are_declared()
    test_facade_remains_replaceable()
    test_cost_model_api_has_no_mlir_or_kernel_policy_dependency()
    test_cost_model_build_owns_the_facade()
    test_api_surface_does_not_expose_mutation()
    print("Cost-model API skeleton source contract: PASS")
