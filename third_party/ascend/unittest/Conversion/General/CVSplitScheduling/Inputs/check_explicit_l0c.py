"""Structural, fallback and independence tests for default L0C materialization."""
from pathlib import Path
import re
import subprocess
import sys
from check_l0c_storage import APPLIED, SSA, live_depths, run, scope, without_ssa

MARKER = "triton_ascend.cv_split_scheduling.explicit_l0c_applied = 1"


def check_storage(ir, lanes):
    assert MARKER in ir and APPLIED in ir
    assert "cv_split.origin_id" not in ir and "cv_split.l0c_lane" not in ir
    cube = scope(ir, "CUBE")
    matrices = re.findall(r"(" + SSA + r") = linalg.matmul .*outs\((" + SSA + r") :", cube)
    assert len(matrices) == 2 * lanes
    allocations = [line for line in cube.splitlines() if "memref.alloc()" in line and "address_space<cc>" in line]
    assert len(allocations) == 4 and all("alignment = 64 : i64" in line for line in allocations)
    tensors = re.findall(r"(" + SSA + r") = bufferization.to_tensor (" + SSA +
                         r") restrict writable : memref<[^\n]+address_space<cc>>", cube)
    assert len(tensors) == 4 and len({buffer for _, buffer in tensors}) == 4
    tensors = dict(tensors)
    fills = dict(re.findall(r"(" + SSA + r") = linalg.fill ins\([^\n]+outs\((" + SSA + r") :", cube))
    destinations = [tensors[fills[initial]] for _, initial in matrices]
    for family in (destinations[:lanes], destinations[lanes:]):
        assert len(set(family)) == 2
        assert all(family[lane] == family[lane % 2] for lane in range(lanes))
    assert set(destinations[:lanes]).isdisjoint(destinations[lanes:])
    first_matrix = cube.index(" = linalg.matmul ")
    inner_loop = cube.rfind("scf.for ", 0, first_matrix)
    assert inner_loop >= 0
    for line in allocations:
        assert cube.index(line) < inner_loop
    for result, initial in matrices:
        fill_position = cube.index(initial + " = linalg.fill ")
        assert inner_loop < fill_position < cube.index(result + " = linalg.matmul ")


def compare(tool, source, mode, lanes=4):
    default = run(tool, source, mode=mode, unroll=lanes)
    explicit = run(tool, source, mode=mode, unroll=lanes, storage="explicit")
    backend = run(tool, source, mode=mode, unroll=lanes, storage="backend")
    assert default == explicit, 'explicit storage is not the default'
    check_storage(explicit, lanes)
    assert without_ssa(scope(backend, 'VECTOR')) == without_ssa(scope(explicit, 'VECTOR'))
    old_allocations = re.findall(r'memref.alloc\(\).*', backend)
    new_allocations = [line for line in re.findall(r'memref.alloc\(\).*', explicit)
                       if 'address_space<cc>' not in line]
    assert old_allocations == new_allocations
    assert re.findall(r'hivm.hir.sync_block_[^\n]+', backend) == re.findall(r'hivm.hir.sync_block_[^\n]+', explicit)
    assert live_depths(backend, lanes) == live_depths(explicit, lanes) == [1, 1]
    print(f'EXPLICIT_STORAGE=PASS mode={mode} lanes={lanes}')


tool, fixture = sys.argv[1:]
source = Path(fixture).read_text()
compare(tool, source, 'disabled')
compare(tool, source, 'materialize')
compare(tool, source, 'disabled', lanes=2)
narrow = Path(fixture).with_name('cv_split_scheduling_fa_m64_n128_seq4096.mlir').read_text()
compare(tool, narrow, 'disabled')
for candidate in (narrow, source.replace('128', '256')):
    ordinary = run(tool, candidate, mode='materialize', storage='backend')
    explicit = run(tool, candidate, mode='materialize')
    assert MARKER not in explicit and explicit == ordinary
nonzero = source.replace('arith.constant 0.000000e+00 : f32', 'arith.constant 1.000000e+00 : f32')
negative_zero = source.replace('arith.constant 0.000000e+00 : f32', 'arith.constant -0.000000e+00 : f32')
for initial in (nonzero, negative_zero):
    ordinary = run(tool, initial, storage='backend')
    explicit = run(tool, initial)
    assert MARKER not in explicit and ordinary == explicit
invalid = subprocess.run([tool, '--cv_split_scheduling=compile-on-910-95=true l0c-buffer-mode=invalid'],
                         input=source, text=True, capture_output=True)
assert invalid.returncode != 0 and 'invalid l0c-buffer-mode' in invalid.stderr
print('EXPLICIT_FALLBACK_AND_INVALID_OPTION=PASS')
