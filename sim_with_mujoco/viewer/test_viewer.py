import time

import glfw
import mujoco

from sim.model.motion.trajectory import interpolate_position
from sim_with_mujoco.utils.ik import solve_ik
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.utils.mj import actuator_ids_from_joints
from sim_with_mujoco.viewer.glfw_panel import GlfwTargetPanel

XML_PATH = "/Users/woojyelee/workspace/my_robotics/assets/robots/robotis_ffw/scene_ffw_sh5.xml"

# # 1. GLFW window / OpenGL context 생성
# window = glfw.create_window(1200, 900, "Demo", None, None)
# glfw.make_context_current(window)

# # 2. MuJoCo visualization objects
# cam = mujoco.MjvCamera()
# opt = mujoco.MjvOption()
# scene = mujoco.MjvScene(model, maxgeom=10000)
# context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# # 3. 매 프레임
# mujoco.mj_step(model, data)

# viewport = mujoco.MjrRect(0, 0, width, height)

# mujoco.mjv_updateScene(
#     model,
#     data,
#     opt,
#     None,
#     cam,
#     mujoco.mjtCatBit.mjCAT_ALL,
#     scene,
# )

# mujoco.mjr_render(viewport, scene, context)

# glfw.swap_buffers(window)
# glfw.poll_events()


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    # model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    initial_qpos = {
        "lift_joint": -0.15,
        "head_joint1": 0.0,
        "head_joint2": 0.0,
        "arm_l_joint1": 0.0,
        "arm_l_joint2": 0.0,
        "arm_l_joint3": 0.0,
        "arm_l_joint4": -1.57,
        "arm_l_joint5": 0.0,
        "arm_l_joint6": 0.0,
        "arm_l_joint7": 0.0,
        "arm_r_joint1": 0.0,
        "arm_r_joint2": 0.0,
        "arm_r_joint3": 0.0,
        "arm_r_joint4": -1.57,
        "arm_r_joint5": 0.0,
        "arm_r_joint6": 0.0,
        "arm_r_joint7": 0.0,
        "finger_l_joint1": 0.3,
        "finger_l_joint2": 1.57,
        "finger_l_joint3": -0.35,
        "finger_l_joint4": -0.25,
        "finger_r_joint1": 0.3,
        "finger_r_joint2": -1.57,
        "finger_r_joint3": 0.35,
        "finger_r_joint4": 0.25,
    }
    for name, value in initial_qpos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        qadr = model.jnt_qposadr[joint_id]
        data.qpos[qadr] = value
        data.ctrl[actuator_id] = data.qpos[qadr]

    # mujoco.mj_kinematics(model, data)  # 현재 qpos 기준으로 body/site/geom xpos, xmat 갱신 (위치/자세 계산 중심)
    mujoco.mj_forward(model, data)  # qpos/qvel/ctrl 기준으로 kinematics + velocity, force, qacc 등등까지 계산

    slider_range = (-0.2, 0.2)  # end-effector 조작 범위
    rotation_slider_range = (-0.15, 0.15)
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "arm_r_link7",
    )
    ik_joint_names = [
        "lift_joint",
        "arm_r_joint1",
        "arm_r_joint2",
        "arm_r_joint3",
        "arm_r_joint4",
        "arm_r_joint5",
        "arm_r_joint6",
        "arm_r_joint7",
    ]
    initial_target_pos = data.xpos[body_id].copy()
    initial_pose = get_body_T(data, body_id)

    hand_pose_panel = GlfwTargetPanel(
        initial_target_pos,
        slider_range=slider_range,
        rotation_slider_range=rotation_slider_range,
    )

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(1200, 900, "MyJoCo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # 창 갱신을 모니터 refresh 주기에 맞춤

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(
        model,
        mujoco.mjtFontScale.mjFONTSCALE_150,
    )

    mujoco.mjv_defaultOption(opt)

    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 1.0]
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -20

    # 궤적 형성 관련 변수
    trajectory_start = initial_target_pos.copy()
    trajectory_goal = initial_target_pos.copy()
    current_target = initial_target_pos.copy()
    trajectory_start_time = None
    trajectory_duration = 0.05  # poll_interval과 같이 고려

    poll_interval = 0.1
    last_poll_time = 0.0
    sim_steps_per_frame = 8

    try:
        while not glfw.window_should_close(window):  # 렌더링 루프
            glfw.poll_events()

            # 1. rendering 주기 / 2. poll 주기 / 3. 시뮬레이션 주기 / 4. step() 자체 주기(예. model.opt.timestep) 분리
            # 주기 관리: 반복문 / 조건문
            # poll_interval = 0.1
            # last_poll_time = 0.0
            now = time.time()
            polled_target = None

            # if now - last_poll_time >= poll_interval:
            #     polled_target = hand_pose_panel.poll_target(window)  # polling 방식
            #     last_poll_time = now

            polled_target = hand_pose_panel.poll_target(window)  # 매 프레임마다 입력 처리
            # if polled_target is not None and now - last_poll_time >= poll_interval:
            if polled_target is not None:
                trajectory_start = current_target.copy()
                trajectory_goal = polled_target[:3].copy()
                trajectory_start_time = time.time()
                last_poll_time = now

            if trajectory_start_time is not None:
                t = time.time() - trajectory_start_time  # t: time-scaling이 입력으로 받는 궤적 시점

                if t >= trajectory_duration:
                    current_target = trajectory_goal.copy()
                    trajectory_start_time = None
                else:
                    current_target = interpolate_position(
                        trajectory_start,
                        trajectory_goal,
                        trajectory_duration,
                        t,
                    )

            new_target = initial_pose.copy()
            new_target[:3, 3] = current_target

            step_start_time = time.time()

            # while time.time() - step_start_time <= 1.0 / 60.0:  # 시뮬레이션 루프
            for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프
                # target = hand_pose_panel.poll_target(window)
                # target_i = interpolate_position(
                #     trajectory_start, trajectory_goal, trajectory_duration, time.time() - step_start
                # )
                # new_target[:3, 3] = target_i

                q_des, joint_ids = solve_ik(model, data, body_id, new_target, True, ik_joint_names)
                actuator_ids = actuator_ids_from_joints(model, joint_ids)

                for i in range(1):  # 수정 예정
                    # joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    # actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                    # if actuator_id < 0:
                    #     continue

                    # IK solver result로 qpos(kinematic simulation용) 또는 ctrl(dynamic simulation용)을 갱신
                    # 옵션 1: dynamic update
                    for actuator_id in actuator_ids:
                        jid = model.actuator_trnid[actuator_id, 0]
                        qadr = model.jnt_qposadr[jid]
                        data.ctrl[actuator_id] = q_des[qadr]  # ctrl 유형은 관절 종류에 따라 다름(예. force/qpos 등)
                    # 옵션 2: kinematic update (테스트용)
                    # for joint_id in joint_ids:
                    #     qadr = model.jnt_qposadr[joint_id]
                    #     data.qpos[qadr] = q_des[qadr]

                    # 옵션 1
                    mujoco.mj_forward(model, data)
                    data.qfrc_applied[:] = 0.0
                    data.qfrc_applied[:] = data.qfrc_bias
                    mujoco.mj_step(model, data)
                    # 옵션 2
                    # data.qvel[:] = 0.0
                    # mujoco.mj_forward(model, data)

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )

            # 다음 화면을 그려 버퍼에 저장
            mujoco.mjr_render(viewport, scene, context)
            hand_pose_panel.render(window, context)

            # GLFW/OpenGL - double buffering
            # render()가 그린 화면을 비로소 창에 띄움
            glfw.swap_buffers(window)

    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()
