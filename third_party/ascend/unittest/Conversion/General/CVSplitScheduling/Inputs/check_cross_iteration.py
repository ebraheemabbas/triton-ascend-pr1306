"""Check admission and exact transactional rollback across unroll modes."""
from pathlib import Path
import re
import subprocess
import sys


tool, fixture = sys.argv[1:]
pieces = Path(fixture).read_text().split("// -----")
assert len(pieces) == 10
checked = 0
for factor in (2, 4, 8):
    for full in (False, True):
        for piece in pieces:
            name = re.search(r"func.func @([a-z0-9_]+)", piece).group(1)
            source = piece
            if full:
                source = source.replace("%limit = arith.constant 16 : index",
                                        f"%limit = arith.constant {factor} : index")
            baseline = subprocess.run([tool, "--mlir-print-op-generic"],
                                      input=source, text=True, capture_output=True,
                                      check=True)
            result = subprocess.run([
                tool, "--mlir-print-op-generic",
                "--debug-only=cv-split-scheduling",
                f"--cv_split_scheduling=compile-on-910-95=true unroll-factor={factor}"
            ], input=source, text=True, capture_output=True, check=True)
            reject = name.startswith("reject_") and not (
                full and factor == 2 and name == "reject_forwarded_c2v")
            if reject:
                assert "CVSplit cross-iteration pre-check rejected:" in result.stderr
                assert result.stdout == baseline.stdout, (name, factor, full)
            else:
                assert "Cross-iteration consumer pre-check passed" in result.stderr, (
                    name, factor, full, result.stderr)
                assert "CVSplit cross-iteration pre-check rejected:" not in result.stderr
            assert "cv_split.loop_carried_consumers" not in result.stdout
            checked += 1
assert checked == 60
print("CROSS_ITERATION_ADMISSION_ROLLBACK_AND_CLEANUP=PASS cases=60")
