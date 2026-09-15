"""Require actual asymmetric materialization and independent typed buffer pools."""
from pathlib import Path
import re
import sys

from check_l0c_storage import APPLIED, PRESERVE, live_depths, run, scope

EXPLICIT = "triton_ascend.cv_split_scheduling.explicit_l0c_applied = 1"


def check(tool, source, rows, score_width, head_dim):
    ir = run(tool, source, mode="materialize")
    assert APPLIED in ir and PRESERVE in ir and EXPLICIT in ir, "materializer fell back"
    assert live_depths(ir, 4) == [1, 1]
    cube = scope(ir, "CUBE")
    vector = scope(ir, "VECTOR")
    assert 'vector_mode = "simd"' in vector
    assert "linalg.matmul" not in vector
    assert len(re.findall(r" = linalg.matmul\b", cube)) == 8
    cc = re.findall(r"memref.alloc\(\).*: memref<([^\n]+address_space<cc>)>", cube)
    expected = [f"{2 * rows}x{width}xf32, #hivm.address_space<cc>"
                for width in (score_width, score_width, head_dim, head_dim)]
    assert sorted(cc) == sorted(expected), cc
    # Transfer pools live in the common parent of CUBE and VECTOR. Per-lane
    # SIMD scratch inside VECTOR may have the same shape, but is not a pool.
    # Count only shared storage, retaining the exact two-slots-per-role check.
    shared = ir.replace(cube, "", 1).replace(vector, "", 1)
    ub = re.findall(r"memref.alloc\(\).*: memref<([^\n]+address_space<ub>)>", shared)
    for width in set((score_width, head_dim)):
        count = 4 if score_width == head_dim else 2
        assert ub.count(f"{rows}x{width}xf32, #hivm.address_space<ub>") == count, ub
    backend = run(tool, source, mode="materialize", storage="backend")
    assert APPLIED in backend and PRESERVE in backend and EXPLICIT not in backend
    assert re.findall(r"hivm.hir.sync_block_[^\n]+", ir) == re.findall(
        r"hivm.hir.sync_block_[^\n]+", backend)
    print(f"MATERIALIZER=PASS rows={rows} score={score_width} product={head_dim}")


tool, fixture = sys.argv[1:]
path = Path(fixture)
check(tool, path.read_text(), 64, 128, 64)
check(tool, path.with_name("cv_split_l0c_storage.mlir").read_text(), 64, 128, 128)
check(tool, path.with_name("cv_split_scheduling_fa_m64_n128_seq4096.mlir").read_text(), 32, 128, 64)
# Score width 32 cannot use the materializer's 64-element vector chunks.
# Unlike zero *additional* UB budget, this is an unavailable geometry and
# must roll back the complete rewrite, not publish only one core's scopes.
unsupported = path.with_name("cv_split_scheduling_fa.mlir").read_text()
rejected = run(tool, unsupported, mode="materialize")
assert APPLIED not in rejected and PRESERVE not in rejected and EXPLICIT not in rejected
print("UNSUPPORTED_GEOMETRY_ROLLBACK=PASS")
