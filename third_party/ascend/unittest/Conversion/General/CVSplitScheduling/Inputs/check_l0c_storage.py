"""Check explicit-default storage and own-producer publication invariants."""
import pathlib
import re
import subprocess
import sys


SSA = r"%[a-zA-Z0-9_.$]+(?:#[0-9]+)?"
APPLIED = "triton_ascend.cv_split_scheduling.applied = 1"
PRESERVE = "triton_ascend.cv_split_scheduling.preserve_explicit_schedule"


def run(tool, source, candidate=2, unroll=4, mode="disabled", storage=None,
        extra=""):
    options = (f"compile-on-910-95=true unroll-factor={unroll} "
               f"schedule-candidate-id={candidate} "
               f"enable-plan-driven-early-publish=true "
               f"post-split-schedule-mode={mode} {extra}")
    if storage is not None:
        options += f" l0c-buffer-mode={storage}"
    result = subprocess.run([tool, f"--cv_split_scheduling={options}"],
                            input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def scope(ir, side):
    end = re.search(r'(?m)^( *)\} \{hivm.tcore_type = #hivm.tcore_type<' + side + '>', ir)
    assert end is not None
    # Select the outer core scope, not its last nested SIMD region.
    starts = list(re.finditer(r'(?m)^' + end.group(1) + r'scope.scope : \(\) -> \(\) \{',
                              ir[:end.start()]))
    assert starts
    return ir[starts[-1].start():end.start()]


def without_ssa(ir):
    return re.sub(SSA, "%value", ir)


def live_depths(ir, lanes):
    cube = scope(ir, "CUBE")
    pending = set()
    peak = [0, 0]
    roles = {}
    matrices = 0
    drains = 0
    for line in cube.splitlines():
        matrix = re.search(r"(" + SSA + r") = linalg.matmul\b", line)
        drain = re.search(r"hivm.hir.fixpipe .*ins\((" + SSA + r")", line)
        if matrix:
            value = matrix.group(1)
            # CUBE emits the QK phase followed by the PV phase.
            role = int(matrices >= lanes)
            roles[value] = role
            pending.add(value)
            peak[role] = max(peak[role], sum(roles[v] == role for v in pending))
            matrices += 1
        elif drain:
            value = drain.group(1)
            assert value in pending, ("unmatched or duplicated drain", line)
            pending.remove(value)
            drains += 1
    assert matrices == drains == 2 * lanes, (matrices, drains, lanes)
    assert not pending
    return peak


def resources(ir):
    allocations = sorted(re.findall(r"memref.alloc\(\).*memref<[^\n]+", ir))
    events = sorted(re.findall(r"hivm.hir.sync_block_(?:set|wait)[^\n]+", ir))
    return allocations, events


def check_pair(tool, source, mode, unroll=4):
    default = run(tool, source, unroll=unroll, mode=mode)
    explicit = run(tool, source, unroll=unroll, mode=mode, storage="explicit")
    backend = run(tool, source, unroll=unroll, mode=mode, storage="backend")
    assert default == explicit, "explicit pools are not the default"
    assert APPLIED in default and APPLIED in backend, "unexpected CVSplit fallback"
    assert (PRESERVE in default) == (mode == "materialize")
    assert (PRESERVE in backend) == (mode == "materialize")
    assert live_depths(default, unroll) == live_depths(backend, unroll) == [1, 1]
    allocations, events = resources(default)
    assert ([a for a in allocations if "address_space<cc>" not in a], events) == resources(backend)
    assert without_ssa(scope(default, "VECTOR")) == without_ssa(scope(backend, "VECTOR")), \
        "VECTOR transformations changed with the storage policy"
    if mode == "materialize":
        assert 'vector_mode = "simd"' in scope(default, "VECTOR")
        assert "tensor<64xf32>" in default
        assert "#hivm.address_space<cbuf>" in default
    print(f"PASS mode={mode} U={unroll}: explicit default, own-producer drains, vector/UB unchanged")


def main():
    tool, fixture = sys.argv[1:]
    source = pathlib.Path(fixture).read_text()
    check_pair(tool, source, "disabled")
    check_pair(tool, source, "materialize")
    # The narrow fixture exercises a different BM/BN/HD, including HD64.
    narrow = pathlib.Path(fixture).with_name("cv_split_scheduling_fa.mlir").read_text()
    check_pair(tool, narrow, "disabled")
    # U2 fits available flags. Larger unrolls retain the existing resource gates.
    for unroll in (2, 8, 16):
        for candidate in (-1, 0, 1, 2):
            default = run(tool, source, unroll=unroll, candidate=candidate)
            backend = run(tool, source, unroll=unroll, candidate=candidate, storage="backend")
            assert (APPLIED in default) == (APPLIED in backend)
            if APPLIED in default:
                assert live_depths(default, unroll) == live_depths(backend, unroll) == [1, 1]
            else:
                assert default == backend
    for shape in (source, narrow):
        for extra in ("private-buffer-ub-budget-bytes=0",):
            default = run(tool, shape, extra=extra)
            backend = run(tool, shape, storage="backend", extra=extra)
            assert (APPLIED in default) == (APPLIED in backend)
    default = run(tool, narrow, mode="materialize")
    backend = run(tool, narrow, mode="materialize", storage="backend")
    assert APPLIED not in default and APPLIED not in backend and default == backend
    for value in ("true", "false"):
        retired = subprocess.run(
            [tool, f"--cv_split_scheduling=compile-on-910-95=true enable-l0c-drain-widening={value}"],
            input=source, text=True, capture_output=True)
        assert retired.returncode != 0 and "enable-l0c-drain-widening" in retired.stderr
    print("PASS alternate candidates/unrolls and asymmetric materializer fallback")


if __name__ == "__main__":
    main()
