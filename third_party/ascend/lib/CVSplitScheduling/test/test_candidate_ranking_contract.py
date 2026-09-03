#!/usr/bin/env python3
"""Source and policy-contract checks for Stage 8.5 ranking."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"


def read(path: Path) -> str:
    return path.read_text()


def policy_score(raw_ii: int, depths: tuple[int, ...],
                 result_bytes: tuple[int, ...]) -> int:
    assert len(depths) == len(result_bytes)
    extra_live_bytes = sum((depth - 1) * size
                           for depth, size in zip(depths, result_bytes))
    return raw_ii + (extra_live_bytes + 255) // 256


def test_generic_ranker_contract() -> None:
    header = read(INCLUDE / "CostModelCandidateRanking.h")
    source = read(LIB / "CostModelCandidateRanking.cpp")
    calibration = read(LIB / "CVSplitRankingA5Experimental.inc")
    for token in (
            "CVSplitCandidateRankEntry",
            "CVSplitCandidateRanking",
            "rankCostModelCandidates",
            "clearsUncertainty",
    ):
        assert token in header
    for token in (
            "matrixLineageInFlightLimits",
            "matrixLineageResultBytes",
            "checkedAdd",
            "checkedMul",
            "checkedCeilDiv",
            "structurallyDominates",
            "liveResultPressureCycles",
            "policyScoreCycles",
    ):
        assert token in source
    assert "kExperimentalLiveResultPressureBytesPerCycle = 256" in calibration
    assert "kExperimentalMinimumRankingMarginBasisPoints = 200" in calibration
    lowered = source.lower()
    for forbidden in (
            "_attn_fwd",
            "flash_attention",
            "native_0316",
            "head_dim",
            "logicalunrollfactor == 4",
    ):
        assert forbidden not in lowered
    for candidate_id in range(3):
        assert f"candidateid == {candidate_id}" not in lowered
        assert f"candidateid != {candidate_id}" not in lowered


def test_provisional_policy_orders_measured_family() -> None:
    result_bytes = (65536, 65536)
    scores = {
        (1, 1): policy_score(6904, (1, 1), result_bytes),
        (2, 1): policy_score(5830, (2, 1), result_bytes),
        (2, 2): policy_score(5830, (2, 2), result_bytes),
    }
    assert scores[(2, 1)] < scores[(1, 1)]
    assert scores[(2, 1)] < scores[(2, 2)]


def test_diagnostics_rank_without_selecting() -> None:
    diagnostics = read(LIB / "CostModelDiagnostics.cpp")
    pass_source = read(LIB / "CVSplitScheduling.cpp")
    cmake = read(LIB / "CMakeLists.txt")
    for token in (
            "cost-model-rank",
            "cost-model-ranking-summary",
            "margin-bp=",
            "required-bp=",
            "clears-uncertainty=",
            "selection-changed=no",
    ):
        assert token in diagnostics
    assert "rankCostModelCandidates" in diagnostics
    assert "fallbackCandidateId" in pass_source
    assert "CostModelCandidateRanking.cpp" in cmake
    assert "selectCandidate" not in diagnostics


if __name__ == "__main__":
    test_generic_ranker_contract()
    test_provisional_policy_orders_measured_family()
    test_diagnostics_rank_without_selecting()
    print("Stage 8.5 candidate ranking source contract: PASS")
