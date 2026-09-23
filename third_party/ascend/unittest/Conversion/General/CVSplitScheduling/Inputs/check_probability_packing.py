"""Check packing-only selection without accepting fallback as materialization."""
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys

from check_l0c_storage import APPLIED, PRESERVE, run, scope, without_ssa

EXPLICIT = "triton_ascend.cv_split_scheduling.explicit_l0c_applied = 1"


def simd_regions(vector):
    lines = vector.splitlines()
    regions = []
    for start, line in enumerate(lines):
        if not re.search(r"\bscope\.scope\b.*\{", line):
            continue
        depth = 0
        for end in range(start, len(lines)):
            depth += lines[end].count("{") - lines[end].count("}")
            if depth == 0:
                if 'vector_mode = "simd"' in lines[end]:
                    regions.append("\n".join(lines[start:end + 1]))
                break
    return regions


def floating_operations(ir):
    return Counter(re.findall(
        r"\b(?:arith\.(?:addf|subf|mulf|divf|maximumf|truncf)|math\.(?:exp|log)|linalg\.reduce)\b", ir))


def check(tool, source, unroll, rows, width=128):
    direct = run(tool, source, unroll=unroll, mode="materialize",
                 extra="probability-packing=direct_nz")
    nd = run(tool, source, unroll=unroll, mode="materialize",
             extra="probability-packing=nd_transpose")
    default = run(tool, source, unroll=unroll, mode="materialize")
    assert default == direct, "default packing changed"
    assert direct != nd, "packing selector had no effect"
    for ir in (direct, nd):
        assert all(marker in ir for marker in (APPLIED, PRESERVE, EXPLICIT)), "materializer fell back"
    assert without_ssa(scope(direct, "CUBE")) == without_ssa(scope(nd, "CUBE")), "CUBE changed"
    assert re.findall(r"hivm.hir.sync_block_[^\n]+", direct) == re.findall(
        r"hivm.hir.sync_block_[^\n]+", nd), "cross-core events changed"
    direct_vector, nd_vector = scope(direct, "VECTOR"), scope(nd, "VECTOR")
    for ir, vector in ((direct, direct_vector), (nd, nd_vector)):
        shared = ir.replace(scope(ir, "CUBE"), "", 1).replace(vector, "", 1)
        allocations = re.findall(r"memref.alloc\(\)[^\n]+", shared)
        if ir == direct:
            shared_allocations = allocations
        else:
            assert allocations == shared_allocations, "shared UB/L1 pools changed"
    assert floating_operations(direct_vector) == floating_operations(nd_vector), "floating operations changed"
    assert without_ssa("\n".join(line for line in direct_vector.splitlines()
                                if "hivm.hir.copy " in line)) == without_ssa(
        "\n".join(line for line in nd_vector.splitlines() if "hivm.hir.copy " in line)), "L1 copies changed"
    direct_regions, nd_regions = simd_regions(direct_vector), simd_regions(nd_vector)
    assert len(direct_regions) == len(nd_regions), "SIMD region boundaries changed"
    assert "linalg.transpose" not in direct_vector
    assert nd_vector.count("linalg.transpose") == unroll
    # Each transpose must remain INSIDE its producer SIMD region, with NZ output.
    pack_regions = [region for region in nd_regions if "linalg.transpose" in region]
    assert len(pack_regions) == unroll
    for region in pack_regions:
        assert region.count("linalg.transpose") == 1
        assert f"tensor<{rows}x{width}xf16>" in region
        assert f"tensor<{rows}x{width // 16}x16xf16>" in region
        assert f"tensor<{width // 16}x{rows}x16xf16>" in region
        assert "permutation = [1, 0, 2]" in region
    for ir in (direct, nd):
        assert f"memref<{width // 16}x{rows // 16}x16x16xf16" in scope(ir, "VECTOR")
    print(f"PACKING=PASS U{unroll} rows={rows} width={width}")


def main():
    tool, fixture = sys.argv[1:]
    root = Path(fixture).parent
    for name, rows in (("cv_split_asymmetric_materializer.mlir", 64),
                       ("cv_split_l0c_storage.mlir", 64),
                       ("cv_split_scheduling_fa_m64_n128_seq4096.mlir", 32)):
        source = (root / name).read_text()
        for unroll in (2, 4):
            check(tool, source, unroll, rows)
        generic = [run(tool, source, mode="disabled", extra="probability-packing=" + mode)
                   for mode in ("direct_nz", "nd_transpose")]
        assert generic[0] == generic[1], "packing option altered generic mode"
        rejected = [run(tool, source, candidate=-1, mode="materialize",
                        extra="probability-packing=" + mode)
                    for mode in ("direct_nz", "nd_transpose")]
        assert rejected[0] == rejected[1]
        assert all(marker not in rejected[0] for marker in (APPLIED, PRESERVE, EXPLICIT))
    unsupported = (root / "cv_split_scheduling_fa.mlir").read_text()
    for mode in ("direct_nz", "nd_transpose"):
        ir = run(tool, unsupported, mode="materialize", extra="probability-packing=" + mode)
        assert all(marker not in ir for marker in (APPLIED, PRESERVE, EXPLICIT))
    invalid = subprocess.run([tool, "--cv_split_scheduling=probability-packing=invalid"],
                             input="module {}", text=True, capture_output=True)
    assert invalid.returncode != 0
    assert "expected direct_nz or nd_transpose" in invalid.stderr
    print("PACKING_DEFAULT_GENERIC_ROLLBACK_INVALID=PASS")


if __name__ == "__main__":
    main()
