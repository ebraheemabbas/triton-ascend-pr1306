"""Check drain placement independently from vector code and UB ownership."""
import pathlib
import re
import subprocess
import sys


SSA = r"%[a-zA-Z0-9_.$]+(?:#[0-9]+)?"
APPLIED = "triton_ascend.cv_split_scheduling.applied = 1"
PRESERVE = "triton_ascend.cv_split_scheduling.preserve_explicit_schedule"


def run(tool, source, candidate=2, unroll=4, mode="disabled", widening=None,
        extra=""):
    options = (f"compile-on-910-95=true unroll-factor={unroll} "
               f"schedule-candidate-id={candidate} "
               f"enable-plan-driven-early-publish=true "
               f"post-split-schedule-mode={mode} {extra}")
    if widening is not None:
        options += f" enable-l0c-drain-widening={str(widening).lower()}"
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
    on = run(tool, source, unroll=unroll, mode=mode, widening=True)
    off = run(tool, source, unroll=unroll, mode=mode, widening=False)
    assert default == on, "default-on changed the pass output"
    assert APPLIED in on and APPLIED in off, "unexpected CVSplit fallback"
    assert (PRESERVE in on) == (mode == "materialize")
    assert (PRESERVE in off) == (mode == "materialize")
    assert live_depths(on, unroll) == [2, 2]
    assert live_depths(off, unroll) == [1, 1]
    assert resources(on) == resources(off), "UB allocations or event contracts changed"
    assert without_ssa(scope(on, "VECTOR")) == without_ssa(scope(off, "VECTOR")), \
        "VECTOR transformations changed with the drain switch"
    if mode == "materialize":
        assert 'vector_mode = "simd"' in scope(off, "VECTOR")
        assert "tensor<64xf32>" in off
        assert "#hivm.address_space<cbuf>" in off
    print(f"PASS mode={mode} U={unroll}: default=on, live 2/2 -> 1/1; vector/resources unchanged")


def main():
    tool, fixture = sys.argv[1:]
    source = pathlib.Path(fixture).read_text()
    check_pair(tool, source, "disabled")
    check_pair(tool, source, "materialize")
    # The narrow fixture exercises a different BM/BN/HD, including HD64.
    narrow = pathlib.Path(fixture).with_name("cv_split_scheduling_fa.mlir").read_text()
    check_pair(tool, narrow, "disabled")
    # U2 fits available flags. Larger unrolls retain the existing resource gates;
    # disabling widening must neither bypass those gates nor change fallback IR.
    for unroll in (2, 8, 16):
        for candidate in (-1, 0, 1, 2):
            on = run(tool, source, unroll=unroll, candidate=candidate, widening=True)
            off = run(tool, source, unroll=unroll, candidate=candidate, widening=False)
            assert (APPLIED in on) == (APPLIED in off)
            if APPLIED in off:
                assert live_depths(off, unroll) == [1, 1]
                assert resources(on) == resources(off)
            else:
                assert on == off
    for candidate in (-1, 0):
        assert run(tool, source, candidate=candidate, widening=True) == \
            run(tool, source, candidate=candidate, widening=False)
    for shape in (source, narrow):
        for extra in ("private-buffer-ub-budget-bytes=0",):
            on = run(tool, shape, widening=True, extra=extra)
            off = run(tool, shape, widening=False, extra=extra)
            assert (APPLIED in on) == (APPLIED in off)
    on = run(tool, narrow, mode="materialize", widening=True)
    off = run(tool, narrow, mode="materialize", widening=False)
    assert APPLIED not in on and APPLIED not in off and on == off
    print("PASS alternate candidates/unrolls and asymmetric materializer fallback")


if __name__ == "__main__":
    main()
