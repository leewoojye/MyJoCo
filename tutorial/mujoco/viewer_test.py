import os
from pathlib import Path
import platform
import sys
import time

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "assets" / "robots" / "arm_model.xml"


def require_mjpython_on_macos() -> None:
    if platform.system() != "Darwin":
        return

    is_mjpython = (
        Path(sys.executable).name == "mjpython" or "MJPYTHON_BIN" in os.environ
    )
    if not is_mjpython:
        raise SystemExit(
            "mujoco.viewer.launch_passive must be run with mjpython on macOS.\n"
            "Run: ./venv/bin/mjpython docs/mujoco/viewer_test.py"
        )


require_mjpython_on_macos()

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
