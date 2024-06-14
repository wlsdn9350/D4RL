""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random
from scipy.signal import convolve2d
import re
import mujoco_py


WALL = 10
EMPTY = 11
GOAL = 12


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    worldbody.site(name='target_site', pos=[0.0,0.0,0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    AGENT_CENTRIC_RES = 32

    def __init__(
        self,
        maze_spec=U_MAZE,
        reward_type="dense",
        reset_target=False,
        agent_centric_view=False,
        **kwargs
    ):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.agent_centric_view = agent_centric_view
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0, 0.0])

        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(
                np.array(self.reset_locations[0]).astype(self.observation_space.dtype)
            )
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        if self.reward_type == "sparse":
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 2.0 else 0.0
        elif self.reward_type == "dense":
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError("Unknown reward type %s" % self.reward_type)
        done = False
        return ob, reward, done, {}

    def render(self, mode, *args, **kwargs):
        if self.agent_centric_view:
            if "width" not in kwargs and "height" not in kwargs:
                kwargs["width"] = self.AGENT_CENTRIC_RES
                kwargs["height"] = self.AGENT_CENTRIC_RES
        return super().render(mode, *args, **kwargs)

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(
                self.observation_space.dtype
            )
            target_location = reset_location + self.np_random.uniform(
                low=-0.1, high=0.1, size=self.model.nq
            )
        self._target = target_location

    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id("target_site")] = np.array(
            [self._target[0] + 1, self._target[1] + 1, 0.0]
        )

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(
            self.observation_space.dtype
        )
        qpos = reset_location + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location  # + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        if self.agent_centric_view:
            self.viewer.cam.type = mujoco_py.generated.const.CAMERA_TRACKING
            self.viewer.cam.distance = 5.0
        else:
            self.viewer.cam.distance = (
                self.model.stat.extent * 1.0
            )  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.lookat[
            0
        ] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = (
            -90
        )  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis



def compute_sampling_probs(maze_layout, filter, temp):
    probs = convolve2d(maze_layout, filter, "valid")
    return np.exp(-temp * probs) / np.sum(np.exp(-temp * probs))


def sample_2d(probs, rng):
    flat_probs = probs.flatten()
    sample = rng.choice(np.arange(flat_probs.shape[0]), p=flat_probs)
    sampled_2d = np.zeros_like(flat_probs)
    sampled_2d[sample] = 1
    idxs = np.where(sampled_2d.reshape(probs.shape))
    return idxs[0][0], idxs[1][0]


def place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp):
    """Samples wall such that overlap with other walls is minimized (overlap is determined by temperature).
    Also adds one door per wall."""
    size = maze_layout.shape[0]
    sample_vert_hor = 0 if rng.random() < 0.5 else 1
    sample_len = int(
        max(
            (max_len_frac - min_len_frac) * size * rng.random() + min_len_frac * size, 3
        )
    )
    sample_door_offset = rng.choice(np.arange(1, sample_len - 1))

    if sample_vert_hor == 0:
        filter = np.ones((sample_len, 5)) / (5 * sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[0]
        sample_pos2 = middle_idxs[1] + 2

        maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
        maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
    else:
        filter = np.ones((5, sample_len)) / (5 * sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[1]
        sample_pos2 = middle_idxs[0] + 2

        maze_layout[sample_pos2, sample_pos1 : sample_pos1 + sample_len] = 1
        maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
    return maze_layout


def sample_layout(
    seed=None, size=20, max_len_frac=0.5, min_len_frac=0.3, coverage_frac=0.25, temp=20
):
    """
    Generates maze layout with randomly placed walls.
    :param seed: if not None, makes maze layout reproducible
    :param size: number of cells per side in maze
    :param max_len_frac: maximum length of walls, as fraction of total maze side length
    :param min_len_frac: minimum length of walls, as fraction of total maze side length
    :param coverage_frac: fraction of cells that is covered with walls in randomly generated layout
    :param temp: controls overlap of walls in maze, the higher the temp the less the overlap of walls
    :return: layout matrix (where 1 indicates wall, 0 indicates free space)
    """
    rng = np.random.default_rng(seed=seed)
    maze_layout = np.zeros((size, size))

    while np.mean(maze_layout) < coverage_frac:
        maze_layout = place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp)

    return maze_layout


def layout2str(layout):
    """Transfers a layout matrix to string format that is used by MazeEnv class."""
    h, w = layout.shape
    padded_layout = np.ones((h + 2, w + 2))
    padded_layout[1:-1, 1:-1] = layout
    output_str = ""
    for row in padded_layout:
        for cell in row:
            output_str += "O" if cell == 0 else "#"
        output_str += "\\"
    output_str = output_str[:-1]  # remove last line break
    output_str = re.sub("O", "G", output_str, count=1)  # add goal at random position
    return output_str


def rand_layout(seed=None, **kwargs):
    """Generates random layout with specified params (see 'sample_layout' function)."""
    rand_layout = sample_layout(seed, **kwargs)
    layout_str = layout2str(rand_layout)
    return layout_str
