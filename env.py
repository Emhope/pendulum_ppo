import numpy as np
import mujoco
from utils import tolerance


class InvertedPendulumEnv:
    xml_env = """
    <mujoco model="inverted pendulum">
            <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="160" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        </asset>
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="1" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.0 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-3 3"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
            <!--geom name="ground" type="plane" pos="0 0 0" /-->
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule" group="3"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-2 2" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-100000 100000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
        </actuator>
    </mujoco>
    """

    def __init__(
        self,
    ):
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.model = mujoco.MjModel.from_xml_string(InvertedPendulumEnv.xml_env)
        self.data = mujoco.MjData(self.model)
        self.traj_len = 1024
        # self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.reset_model()

    def step(self, a):
        self.data.ctrl = a
        mujoco.mj_step(self.model, self.data)
        # self.viewer.sync()
        ob = self.obs()
        pole_angle_cosine = np.cos(ob[1])
        reward = self._get_reward(
            pole_angle_cosine=pole_angle_cosine,
            cart_position=ob[0],
            control=a,
            angular_vel=ob[3],
            ball_pos=self.ball_pos
        )
        self.steps += 1

        # создание новой цели, если текущая достигнута
        if abs(ob[0]-self.ball_pos) < 0.05 and pole_angle_cosine > 0.9:
            #print('ball reached')
            self.ball_pos = np.random.uniform(-1, 1)
        terminated = bool(not np.isfinite(ob).all()) or (self.steps > self.traj_len)# or (pole_angle_cosine > 0.9 and abs(ob[3]) < 0.1)
        if abs(ob[0]) > 1.99:
            terminated = True
            reward -= 1
        return ob, reward, terminated
    
    def _get_reward(self, pole_angle_cosine, cart_position, control, angular_vel, ball_pos):
        upright = (pole_angle_cosine + 1) / 2
        centered = tolerance(cart_position, margin=2)
        centered = (1 + centered) / 2
        small_control = tolerance(control, margin=1, value_at_margin=0, gauss_sigm=False)[0]
        small_control = (4 + small_control) / 5
        small_velocity = tolerance(angular_vel, margin=5)#.min()
        small_velocity = (1 + small_velocity) / 2
        return upright.mean() * small_control * small_velocity * centered #+ abs(ball_pos - cart_position)
    
    def obs(self):
        """
        obs: [cart_pos, pole_angle, cart_vel, pole_vel]
        """
        #return np.concatenate([self.data.qpos, self.data.qvel, np.array([self.ball_pos])]).ravel()
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def reset_model(self):
        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel
        self.data.qpos[1] = 3.14  # Set the pole to be facing down
        self.steps = 0
        self.ball_pos = np.random.uniform(-1, 1)
        return self.obs()

    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def draw_ball(self, color=[1, 0, 0, 1], radius=0.01):
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=np.array(np.array([self.ball_pos, 0, 0.6])),
            mat=np.eye(3).flatten(),
            rgba=np.array(color),
        )
        self.viewer.user_scn.ngeom = 1

    @property
    def current_time(self):
        return self.data.time