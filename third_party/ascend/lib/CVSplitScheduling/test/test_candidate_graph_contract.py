#!/usr/bin/env python3
"""Source-contract checks for Candidate graphs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"


def read(path):
    return path.read_text()


def test_owned_graph_and_generic_edges():
    header = read(INCLUDE / "CostModelCandidateGraph.h")
    source = read(LIB / "CostModelCandidateGraph.cpp")
    for token in ("CVSplitOwnedScheduleRequest", "CVSplitCostedNode", "CVSplitCostedEdge", "getRequest() const",
                  "matrixLineageInFlightLimits", "matrixLineageResultBytes"):
        assert token in header
    for token in ("candidate.logicalLaneCount", "lineage.phaseOrdinal", "lineage.inFlightLimit", "c2vPhases",
                  "EventGenerateWait", "CrossCoreAvailability", "ResourceSerialization", "StorageReuse", "LoopCarried"):
        assert token in source
    lowered = source.lower()
    for token in ("_attn_fwd", "flash_attention", "native_0316", "head_dim", "lane3", "candidateid ==",
                  "logicalunrollfactor == 4"):
        assert token not in lowered


def test_diagnostic_schedule_never_selects():
    diagnostics = read(LIB / "CostModelDiagnostics.cpp")
    pass_source = read(LIB / "CVSplitScheduling.cpp")
    assert "buildCostModelCandidateGraphs" in diagnostics
    assert "estimateSchedule" in diagnostics
    assert "cost-model-schedule-summary" in diagnostics
    assert "selection-changed=no" in diagnostics
    assert "logCandidateScheduleEstimates" in pass_source
    assert "selectCandidate" not in diagnostics


def test_build_owns_graph():
    assert "CostModelCandidateGraph.cpp" in read(LIB / "CMakeLists.txt")


if __name__ == "__main__":
    test_owned_graph_and_generic_edges()
    test_diagnostic_schedule_never_selects()
    test_build_owns_graph()
    print("Candidate-graph source contract: PASS")
