import mujoco
import mujoco.viewer
import time

# 1. 모델 로드 (파일 경로 확인!)
model = mujoco.MjModel.from_xml_path('arm_model.xml')
data = mujoco.MjData(model)

# 2. 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 물리 연산 수행
        mujoco.mj_step(model, data)

        # 화면 갱신
        viewer.sync()

        # 시뮬레이션 속도 조절 (실시간 유지)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)