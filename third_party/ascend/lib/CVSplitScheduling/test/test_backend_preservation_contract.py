#!/usr/bin/env python3
"""Source-contract checks for Stage 9.1 backend preservation."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ATTRIBUTES = ROOT / "third_party/ascend/include/CVSplitScheduling/Attributes.h"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PASS = ROOT / "third_party/ascend/lib/CVSplitScheduling/CVSplitScheduling.cpp"

ATTRIBUTE = "triton_ascend.cv_split_scheduling.preserve_explicit_schedule"


def read(path: Path) -> str:
    return path.read_text()


def test_attribute_and_metadata_contract() -> None:
    attributes = read(ATTRIBUTES)
    backend = read(BACKEND)
    assert "kPreserveExplicitScheduleAttr" in attributes
    assert ATTRIBUTE in attributes
    assert "PRESERVE_EXPLICIT_CV_SPLIT_SCHEDULE_REGEX" in backend
    assert 'metadata["cv_split_preserve_explicit_schedule"]' in backend


def test_preservation_overrides_conflicting_user_policy() -> None:
    backend = read(BACKEND)
    auto_bind = backend.split("def get_auto_bind_sub_block_option", 1)[1]
    auto_bind = auto_bind.split("def get_graph_sync_solver_option", 1)[0]
    graph_sync = backend.split("def get_graph_sync_solver_option", 1)[1]
    graph_sync = graph_sync.split("def _save_npuir_debug_output", 1)[0]
    assert "_preserves_explicit_cv_split_schedule(metadata)" in auto_bind
    assert "return False" in auto_bind
    assert "_preserves_explicit_cv_split_schedule(metadata)" in graph_sync
    assert "return True" in graph_sync
    assert backend.count(
        "sync_solver = get_graph_sync_solver_option(metadata)") == 2


def test_option_helper_semantics() -> None:
    tree = ast.parse(read(BACKEND).lstrip("\ufeff"))
    names = {
        "_preserves_explicit_cv_split_schedule",
        "get_auto_bind_sub_block_option",
        "get_graph_sync_solver_option",
    }
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 str(BACKEND), "exec"), namespace)

    metadata = {
        "auto_tile_and_bind_subblock": False,
        "enable_auto_bind_sub_block": True,
        "sync_solver": False,
    }
    assert namespace["get_auto_bind_sub_block_option"](metadata) is True
    assert namespace["get_graph_sync_solver_option"](metadata) is False
    metadata["cv_split_preserve_explicit_schedule"] = True
    assert namespace["get_auto_bind_sub_block_option"](metadata) is False
    assert namespace["get_graph_sync_solver_option"](metadata) is True


def test_preservation_is_emitted_only_by_stage94_atomic_publication() -> None:
    pass_source = read(PASS)
    backend = read(BACKEND)
    atomic_marker = "if (enableStage94AtomicRewrite) {"
    assert atomic_marker in pass_source
    before_atomic, atomic_and_after = pass_source.split(atomic_marker, 1)
    assert "kPreserveExplicitScheduleAttr" not in before_atomic
    assert "kPreserveExplicitScheduleAttr" in atomic_and_after
    assert pass_source.count("kPreserveExplicitScheduleAttr") == 1
    assert 'metadata["enable_auto_bind_sub_block"]' in backend
    assert 'return metadata["sync_solver"]' in backend


if __name__ == "__main__":
    test_attribute_and_metadata_contract()
    test_preservation_overrides_conflicting_user_policy()
    test_option_helper_semantics()
    test_preservation_is_emitted_only_by_stage94_atomic_publication()
    print("Stage 9.1 backend preservation source contract: PASS")
