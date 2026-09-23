import json
import os
import sys
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest
import triton.backends.ascend.compiler as compiler

pytestmark = pytest.mark.backend("cpu")


def test_cv_split_preserves_user_pipeline_policy():
    metadata = {
        "multibuffer": True,
        "set_workspace_multibuffer": 2,
        "has_auto_blockify_blacklist_op": False,
        "enable_mixed_cv": False,
        "disable_auto_inject_block_sync": False,
    }

    compiler._configure_cv_split_metadata(metadata)

    assert metadata["multibuffer"] is True
    assert metadata["set_workspace_multibuffer"] == 2
    assert metadata["has_auto_blockify_blacklist_op"] is False
    assert metadata["enable_mixed_cv"] is False
    assert metadata["disable_auto_inject_block_sync"] is True


@pytest.mark.parametrize(
    "dynamic, cv_split, target_is_a5, expected",
    [
        (False, False, True, (False, False)),
        (True, False, True, (False, True)),
        (False, True, True, (True, False)),
        # Auto mode: attempt CV split, then let DCVP observe the commit result
        # and run only if CV split kept the original module.
        (True, True, True, (True, True)),
        (True, True, False, (False, True)),
    ],
)
def test_cv_pipeline_selection(dynamic, cv_split, target_is_a5, expected):
    metadata = {
        "enable_dynamic_cv_pipeline": dynamic,
        "enable_cv_split_scheduling": cv_split,
    }
    assert compiler._select_cv_pipeline_policy(metadata, target_is_a5) == expected


def test_cv_split_defaults_are_opt_in_with_materialization():
    """CVSplit opt-in is separate from its default materialization policy."""
    fields = compiler.NPUOptions.__dataclass_fields__
    assert fields["enable_cv_split_scheduling"].default is False
    assert fields["enable_dynamic_cv_pipeline"].default is None
    assert fields["cv_split_unroll_factor"].default == 4
    defaults = compiler.NPUOptions()
    assert defaults.enable_cv_split_scheduling is False
    assert defaults.enable_dynamic_cv_pipeline is None
    assert defaults.cv_split_post_split_schedule_mode == "materialize"
    assert defaults.cv_split_enable_plan_driven_early_publish is True
    assert defaults.cv_split_schedule_candidate_id == -1
    assert defaults.cv_split_l0c_buffer_mode == "explicit"
    assert defaults.limit_auto_multi_buffer_of_local_buffer is None
    assert compiler.NPUOptions(enable_cv_split_scheduling=True).enable_cv_split_scheduling is True
    assert compiler.NPUOptions(enable_cv_split_scheduling=False).enable_cv_split_scheduling is False
    assert compiler.NPUOptions(enable_dynamic_cv_pipeline=True).enable_dynamic_cv_pipeline is True


_UNSET = object()


@pytest.mark.parametrize("arch,is_a5", [("Ascend910B1", False), ("Ascend910_9589", True),
                                       ("Ascend950PR_9579", True)])
@pytest.mark.parametrize("requested", [_UNSET, None, False, True])
def test_cv_split_parse_options_requires_opt_in(arch, is_a5, requested):
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch=arch))
    opts = {} if requested is _UNSET else {"enable_cv_split_scheduling": requested}
    result = backend.parse_options(opts)
    assert result.enable_cv_split_scheduling is (requested is True)
    assert result.enable_dynamic_cv_pipeline is is_a5
    assert result.cv_split_post_split_schedule_mode == "materialize"
    assert result.cv_split_enable_plan_driven_early_publish is True
    assert result.cv_split_schedule_candidate_id == -1


def test_cv_split_explicit_policy_overrides_survive_parse_options():
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend950PR_9579"))
    requested = dict(enable_cv_split_scheduling=True, enable_dynamic_cv_pipeline=False,
                     cv_split_post_split_schedule_mode="disabled",
                     cv_split_enable_plan_driven_early_publish=False,
                     cv_split_schedule_candidate_id=2, cv_split_l0c_buffer_mode="backend",
                     limit_auto_multi_buffer_of_local_buffer="no-limit")
    result = backend.parse_options(dict(requested))
    assert all(getattr(result, name) == value for name, value in requested.items())


@pytest.mark.parametrize("explicit_applied,preserve", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("requested", [_UNSET, None, "no-l0c", "no-limit", "no-ub"])
def test_cv_split_local_multibuffer_policy_preserves_user_choice(explicit_applied, preserve, requested):
    metadata = dict(cv_split_l0c_buffers_applied=explicit_applied,
                    cv_split_preserve_explicit_schedule=preserve)
    if requested is not _UNSET:
        metadata["limit_auto_multi_buffer_of_local_buffer"] = requested
    original = dict(metadata)
    automatic = "no-l0c" if explicit_applied and preserve else "no-limit"
    expected = automatic if requested is _UNSET or requested is None else requested
    assert compiler.get_auto_multi_buffer_of_local_buffer_option(metadata) == expected
    assert metadata == original, "resolving effective policy must not overwrite user options"


@pytest.mark.parametrize("metadata", [{}, {"cv_split_l0c_buffers_applied": True},
                                      {"cv_split_preserve_explicit_schedule": True},
                                      {"enable_cv_split_scheduling": True,
                                       "cv_split_post_split_schedule_mode": "materialize",
                                       "cv_split_l0c_buffer_mode": "explicit"}])
def test_cv_split_requested_policy_is_not_proof_of_application(metadata):
    assert compiler.get_auto_multi_buffer_of_local_buffer_option(metadata) == "no-limit"


@pytest.mark.parametrize("requested", [None, "no-l0c", "no-limit", "no-ub"])
@pytest.mark.parametrize("explicit_applied,preserve", [(False, False), (False, True), (True, False), (True, True)])
def test_cv_split_local_multibuffer_bisheng_argument(monkeypatch, requested, explicit_applied, preserve):
    """Exercise the real A5 argv builder without invoking a compiler or NPU."""
    calls = []

    def fake_run(command, **kwargs):
        from pathlib import Path
        calls.append(command)
        Path(command[command.index("-o") + 1] + ".o").write_bytes(b"test-kernel")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(compiler, "NPUUtils", lambda: SimpleNamespace(has_device_limit=lambda: False))
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))
    monkeypatch.setattr(compiler, "_check_bishengir_api_change", lambda: True)
    monkeypatch.setattr(compiler, "_downgrade_ir_for_bishengir", lambda text: text)
    monkeypatch.setattr(compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(compiler, "_is_ascend_sanitizer_enabled", lambda: False)
    monkeypatch.setattr(compiler, "_enable_msdebug", lambda: False)
    monkeypatch.setattr(compiler, "_enable_print_ub_bits", lambda: False)
    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    options = compiler.NPUOptions(arch="Ascend950PR_9579", debug=False,
                                  limit_auto_multi_buffer_of_local_buffer=requested)
    metadata = dict(options.__dict__, hash="test", target=SimpleNamespace(arch=options.arch),
                    cv_split_l0c_buffers_applied=explicit_applied)
    attribute = " attributes {triton_ascend.cv_split_scheduling.preserve_explicit_schedule}" if preserve else ""
    source = ('module' + attribute + ' { func.func @test() attributes '
              '{mix_mode = "mix", parallel_mode = "simd"} { return } }')
    assert compiler.linalg_to_bin_enable_npu_compile_910_95(source, metadata, options) == b"test-kernel"
    assert len(calls) == 1
    prefix = "--limit-auto-multi-buffer-of-local-buffer="
    expected = requested if requested is not None else ("no-l0c" if explicit_applied and preserve else "no-limit")
    assert [arg for arg in calls[0] if arg.startswith(prefix)] == [prefix + expected]
    assert metadata["limit_auto_multi_buffer_of_local_buffer"] == requested


def _make_torch_npu_mock(cfg_dir):
    """Create a mock torch_npu module whose __file__ is in cfg_dir."""
    mock = MagicMock()
    mock.__file__ = os.path.join(cfg_dir, "__init__.py")
    return mock


def test_cv_split_l0c_storage_option_and_cache_key(monkeypatch):
    monkeypatch.setattr(compiler, "get_cann_version_file_hash", lambda: "fixed-toolchain")
    defaults = compiler.NPUOptions()
    assert defaults.cv_split_l0c_buffer_mode == "explicit"
    assert defaults.hash() == compiler.NPUOptions(cv_split_l0c_buffer_mode="explicit").hash()
    backend = compiler.NPUOptions(cv_split_l0c_buffer_mode="backend")
    assert defaults.hash() != backend.hash()


@pytest.mark.parametrize("packing", ["direct_nz", "nd_transpose"])
def test_cv_split_probability_packing_option(monkeypatch, packing):
    monkeypatch.setattr(compiler, "get_cann_version_file_hash", lambda: "fixed-toolchain")
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend950PR_9579"))
    requested = dict(enable_cv_split_scheduling=True, enable_dynamic_cv_pipeline=False,
                     cv_split_schedule_candidate_id=2,
                     cv_split_probability_packing=packing)
    options = backend.parse_options(requested)
    assert options.cv_split_probability_packing == packing
    assert options.cv_split_post_split_schedule_mode == "materialize"
    assert options.cv_split_l0c_buffer_mode == "explicit"
    assert options.cv_split_schedule_candidate_id == 2
    assert options.enable_dynamic_cv_pipeline is False
    default = compiler.NPUOptions()
    direct = compiler.NPUOptions(cv_split_probability_packing="direct_nz")
    transposed = compiler.NPUOptions(cv_split_probability_packing="nd_transpose")
    assert default.cv_split_probability_packing == "direct_nz"
    assert default.hash() == direct.hash()
    assert direct.hash() != transposed.hash()
    # Selecting a representation must not implicitly enable CVSplit.
    assert compiler.NPUOptions(cv_split_probability_packing=packing).enable_cv_split_scheduling is False


@pytest.mark.parametrize("packing", [None, "", "nz", "ND_TRANSPOSE", "invalid", 0])
def test_cv_split_probability_packing_rejects_invalid_values(packing):
    with pytest.raises(ValueError, match="cv_split_probability_packing"):
        compiler.NPUOptions(cv_split_probability_packing=packing)


@pytest.mark.parametrize("applied", [True, False])
def test_cv_split_l0c_application_metadata(monkeypatch, applied):
    marker = "triton_ascend.cv_split_scheduling.explicit_l0c_applied"
    monkeypatch.setattr(compiler, "_get_then_remove_rc",
                        lambda mod, name: int(applied) if name == marker else -1)
    metadata = {}
    compiler._adjust_metadata_by_module_result(
        None, metadata, compiler.NPUOptions())
    assert metadata["cv_split_l0c_buffers_applied"] is applied
    ordinary = {}
    compiler._adjust_metadata_by_module_result(
        None, ordinary, compiler.NPUOptions(cv_split_l0c_buffer_mode="backend"))
    assert "cv_split_l0c_buffers_applied" not in ordinary


@pytest.mark.parametrize("value", [True, False])
def test_removed_l0c_widening_option_is_rejected(value):
    assert "cv_split_enable_l0c_drain_widening" not in compiler.NPUOptions.__dataclass_fields__
    with pytest.raises(TypeError, match="cv_split_enable_l0c_drain_widening"):
        compiler.NPUOptions(cv_split_enable_l0c_drain_widening=value)


def _write_acl_config(cfg_dir, config):
    """Write acl_default.json into cfg_dir."""
    cfg_path = os.path.join(cfg_dir, "acl_default.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    return cfg_path


def test_get_simt_stack_limit_default(monkeypatch):
    """Without torch_npu installed, should return default 1152."""
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_from_config(monkeypatch, tmp_path):
    """When acl_default.json has valid simt_stack_size, return it."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 2048}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 2048


def test_get_simt_stack_limit_no_StackSize_key(monkeypatch, tmp_path):
    """When acl_default.json has no StackSize key, return default 1152."""
    _write_acl_config(str(tmp_path), {"OtherConfig": {"foo": 1}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_no_simt_stack_size_key(monkeypatch, tmp_path):
    """When StackSize exists but simt_stack_size is absent, return default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {"other_field": 100}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_empty_StackSize(monkeypatch, tmp_path):
    """When StackSize is an empty dict, return default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_empty_config(monkeypatch, tmp_path):
    """When acl_default.json is an empty dict, return default 1152."""
    _write_acl_config(str(tmp_path), {})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_file_not_found(monkeypatch, tmp_path):
    """When acl_default.json does not exist, return default 1152."""
    # Do not write acl_default.json; only mock torch_npu.__file__
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_invalid_json(monkeypatch, tmp_path):
    """When acl_default.json is invalid JSON, return default 1152."""
    cfg_path = os.path.join(str(tmp_path), "acl_default.json")
    with open(cfg_path, "w") as f:
        f.write("{invalid json content}")

    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_from_config_returns_int(monkeypatch, tmp_path):
    """Return value from config should be an integer."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 2048}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert isinstance(result, int)
    assert result == 2048


def test_get_simt_stack_limit_config_overrides_default(monkeypatch, tmp_path):
    """Config value should take precedence over default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 9999}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 9999
    assert result != 1152
