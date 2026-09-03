from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PYBIND = ROOT / "third_party/ascend/triton_ascend.cc"


def read(path):
    return path.read_text()


def test_stage71_options_are_default_off_and_propagated():
    passes = read(INCLUDE / "Passes.td")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    backend = read(BACKEND)
    pybind = read(PYBIND)

    assert 'Option<"enablePurePrerequisiteHoisting"' in passes
    assert '"bool", /*default*/"false"' in passes
    assert 'Option<"purePrerequisiteHoistBudgetBytes"' in passes
    assert '"int64_t", /*default*/"0"' in passes
    assert "options.enablePurePrerequisiteHoisting" in cpp
    assert "options.purePrerequisiteHoistBudgetBytes" in cpp
    assert "opts.enablePurePrerequisiteHoisting" in pybind
    assert "opts.purePrerequisiteHoistBudgetBytes" in pybind
    assert "cv_split_enable_pure_prerequisite_hoisting: bool = False" in backend
    assert "cv_split_pure_prerequisite_hoist_budget_bytes: int = 0" in backend
    assert 'metadata["cv_split_enable_pure_prerequisite_hoisting"]' in backend
    assert 'metadata["cv_split_pure_prerequisite_hoist_budget_bytes"]' in backend


def test_stage71_uses_transfer_and_ssa_facts_only():
    header = read(INCLUDE / "PurePrerequisiteHoisting.h")
    cpp = read(LIB / "PurePrerequisiteHoisting.cpp")
    transfers = read(INCLUDE / "CrossScopeTransfers.h")

    assert "CubeToVectorTransferChain" in transfers
    assert "Operation *wait" in transfers
    assert "Value transferredValue" in transfers
    assert "collectSameBlockPredecessors" in cpp
    assert "collectSameBlockDescendants" in cpp
    assert "futureConsumerSlice" in cpp
    assert "collectSameBlockPredecessors(firstConsumer" not in cpp
    assert "futureIndex = chainIndex" in cpp
    assert "chains[futureIndex]->consumers" in cpp
    assert "isMemoryEffectFree" in cpp
    assert "closeMovableOperands" in cpp
    assert "EngineType::VECTOR" in cpp
    assert "getStaticScalarOrRankOneBytes" in cpp
    assert "getAdditionalLiveBytes" in cpp
    assert "op->moveBefore(wait)" in cpp
    assert "hoistPurePrerequisites" in header

    forbidden = (
        "softmax",
        "alpha",
        "qk",
        "pv",
        "head_dim",
        "kernel",
        "lane3",
        "flag == 8",
        "unrollFactor == 4",
    )
    lowered = cpp.lower()
    for token in forbidden:
        assert token not in lowered


def test_stage71_rejects_effects_rank2_and_unavailable_operands():
    cpp = read(LIB / "PurePrerequisiteHoisting.cpp")

    assert "op->getNumRegions() != 0" in cpp
    assert "op->getNumResults() == 0" in cpp
    assert "tensorType.getRank() > 1" in cpp
    assert "!isMemoryEffectFree(op)" in cpp
    assert "!selected.contains(def)" in cpp
    assert "budget-rejected" in cpp
    assert "budgetBytes == 0" in cpp
    assert "std::numeric_limits<uint64_t>::max()" in cpp
    assert "MulOverflow" not in cpp
    assert "AddOverflow" not in cpp


def test_stage71_runs_after_transfer_emission_before_scope_split():
    cpp = read(LIB / "CVSplitScheduling.cpp")
    transfer = cpp.index("insertCrossScopeTransfers(")
    hoist = cpp.index("hoistPurePrerequisites(")
    remove_origins = cpp.index("removeUnrollOriginIdAttrs(funcOp)", transfer)
    scope = cpp.index("createScopeSeparation(", transfer)

    assert transfer < hoist < remove_origins < scope


def test_stage71_build_owns_new_implementation():
    cmake = read(LIB / "CMakeLists.txt")
    assert "PurePrerequisiteHoisting.cpp" in cmake


if __name__ == "__main__":
    test_stage71_options_are_default_off_and_propagated()
    test_stage71_uses_transfer_and_ssa_facts_only()
    test_stage71_rejects_effects_rank2_and_unavailable_operands()
    test_stage71_runs_after_transfer_emission_before_scope_split()
    test_stage71_build_owns_new_implementation()
    print("Stage 7.1 pure-prerequisite hoisting source contract: PASS")
