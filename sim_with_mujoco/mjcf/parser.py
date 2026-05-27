import mujoco
import numpy as np
from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parents[3]
# XML_PATH = ROOT_DIR / "assets" / "robots" / "robotis_ffw" / "scene_ffw_sh5.xml"


def parser(xml_path):
    # 1. MuJoCo로 XML 파싱 및 초기 구조 불러오기
    # XML 파일을 읽어와 MjModel 객체를 생성
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    # MjModel과 달리 동적인 현재 상태를 담는 객체로, qpos, qvel 등 정보를 담음
    data = mujoco.MjData(model)
    # data.qpos[:] = data.qpos

    # 초기 상태의 절대 좌표계를 한 번 계산
    mujoco.mj_kinematics(model, data)

    return model, data
