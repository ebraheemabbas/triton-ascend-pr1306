"""Check materialize/early-publish defaults without selecting a hidden candidate."""
from pathlib import Path
import subprocess
import sys

from check_l0c_storage import APPLIED, PRESERVE

tool, fixture = sys.argv[1:]
root = Path(fixture).parent
EXPLICIT = "triton_ascend.cv_split_scheduling.explicit_l0c_applied = 1"


def run(source, options=None, debug=False):
    command = [tool]
    if options is not None:
        command.append("--cv_split_scheduling=" + options)
    if debug:
        command.append("--debug-only=cv-split-scheduling")
    return subprocess.run(command, input=source, text=True, capture_output=True, check=True)


for name in ("cv_split_asymmetric_materializer.mlir", "cv_split_l0c_storage.mlir"):
    source = (root / name).read_text()
    common = "compile-on-910-95=true unroll-factor=4"
    baseline = run(source).stdout
    unselected = run(source, common, debug=True)
    minus_one = run(source, common + " schedule-candidate-id=-1")
    assert unselected.stdout == minus_one.stdout == baseline, name
    assert "structural-candidate-ready=no" in unselected.stderr, unselected.stderr
    assert all(marker not in unselected.stdout for marker in (APPLIED, PRESERVE, EXPLICIT))

    selected = run(source, common + " schedule-candidate-id=2").stdout
    explicit_policy = run(source, common + " schedule-candidate-id=2 "
                          "post-split-schedule-mode=materialize "
                          "enable-plan-driven-early-publish=true").stdout
    assert selected == explicit_policy, "pass defaults differ from the explicit qualified policy"
    assert all(marker in selected for marker in (APPLIED, PRESERVE, EXPLICIT)), name

    generic = run(source, common + " post-split-schedule-mode=disabled "
                  "enable-plan-driven-early-publish=false").stdout
    assert APPLIED in generic and PRESERVE not in generic, name
    print(f"DEFAULT_POLICY=PASS fixture={name}: candidate=-1 rollback, candidate2 materialization")
