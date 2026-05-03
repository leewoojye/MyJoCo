import mujoco
import mujoco.viewer
import time

# 아주 기본적인 바닥과 상자 모델 (XML)
model_xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 모델 로드
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

# 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)