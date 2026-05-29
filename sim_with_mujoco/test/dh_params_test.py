import mujoco
import numpy as np


L1 = 0.3
L2 = 0.2


XML = f"""
<mujoco model="dh_2r_example">
    <compiler angle="radian"/>
    <option gravity="0 0 0"/>

    <worldbody>
        <body name="link1" pos="0 0 0">
            <joint name="theta1" type="hinge" axis="0 0 1"/>
            <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.01"/>

            <body name="link2" pos="{L1} 0 0">
                <joint name="theta2" type="hinge" axis="0 0 1"/>
                <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.01"/>
                <site name="ee" pos="{L2} 0 0" size="0.02"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""


def dh_transform(a, alpha, d, theta):
    """Standard DH transform: RotZ(theta) TransZ(d) TransX(a) RotX(alpha)."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def dh_forward_kinematics(q):
    dh_params = [
        # a, alpha, d, theta
        (L1, 0.0, 0.0, q[0]),
        (L2, 0.0, 0.0, q[1]),
    ]

    T = np.eye(4)
    for a, alpha, d, theta in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)
    return T


def test_mujoco_fk_matches_dh_fk():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    q = np.deg2rad([30.0, -45.0])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    mujoco_ee_pos = data.site_xpos[ee_id].copy()

    dh_T = dh_forward_kinematics(q)
    dh_ee_pos = dh_T[:3, 3]

    print("MuJoCo ee position:", mujoco_ee_pos)
    print("DH ee position:    ", dh_ee_pos)

    np.testing.assert_allclose(mujoco_ee_pos, dh_ee_pos, atol=1e-12)


if __name__ == "__main__":
    test_mujoco_fk_matches_dh_fk()
