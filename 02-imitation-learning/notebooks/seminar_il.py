"""SO-101 teaching environment for Seminar 02.

This module is intentionally small and notebook-oriented.  MuJoCo advances the
robot and scene, while a documented grasp assist makes data collection reliable
enough for a 90-minute Colab seminar.  The assist only attaches a cube when the
open gripper is correctly aligned and then closes; navigation and obstacle
contacts remain simulated by MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import mujoco
import numpy as np


JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
CUBE_NAMES = ("red_box_joint", "green_box_joint", "blue_box_joint")
CUBE_COLORS = ("red", "green", "blue")

# Match the single-cube obstacle distributions from ETH HW3.
OBSTACLE_POS_STD = 0.010
ADVERSARIAL_OBSTACLE_POS_STD = 0.005
OBSTACLE_SHIFT_X = 0.080
ADVERSARIAL_CENTER_PROB = 0.20


def locate_asset_dir(start: Path | None = None) -> Path:
    """Find the seminar asset directory from common notebook working folders."""
    start = (start or Path.cwd()).resolve()
    module_dir = Path(__file__).resolve().parent
    # Colab executes an opened notebook from /content even when seminar_il.py
    # lives in /content/multimodal-vla-course/02-imitation-learning/notebooks.
    # Search relative to both locations so importing the module does not depend
    # on the process working directory.
    roots = tuple(dict.fromkeys((start, *start.parents, module_dir, *module_dir.parents)))
    relatives = (
        Path("assets/so101_gym"),
        Path("02-imitation-learning/assets/so101_gym"),
    )
    for root in roots:
        for relative in relatives:
            candidate = root / relative
            if (candidate / "so100_transfer_cube_obstacle_ee.xml").is_file():
                return candidate
    raise FileNotFoundError(
        "SO-101 assets were not found. Keep the repository's "
        "02-imitation-learning/assets directory beside the notebook."
    )


@dataclass
class Snapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    time: float
    attached_cube: int | None


class SO101Env:
    """Minimal SO-101 manipulation environment with Cartesian delta control."""

    def __init__(
        self,
        scene: str = "single",
        *,
        obstacle_mode: str = "train",
        seed: int = 0,
        control_hz: float = 20.0,
        render_size: tuple[int, int] = (320, 240),
        asset_dir: Path | None = None,
    ) -> None:
        if scene not in {"single", "multicube"}:
            raise ValueError("scene must be 'single' or 'multicube'")
        if obstacle_mode not in {"train", "adversarial"}:
            raise ValueError("obstacle_mode must be 'train' or 'adversarial'")
        self.scene = scene
        self.obstacle_mode = obstacle_mode
        self.rng = np.random.default_rng(seed)
        self.asset_dir = asset_dir or locate_asset_dir()
        xml_name = (
            "so100_transfer_cube_obstacle_ee.xml"
            if scene == "single"
            else "so100_multicube_ee.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.asset_dir / xml_name))
        self.data = mujoco.MjData(self.model)
        self.control_hz = float(control_hz)
        self.substeps = max(1, round((1.0 / self.control_hz) / self.model.opt.timestep))

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        )
        self.joint_qpos = self.model.jnt_qposadr[self.joint_ids]
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES]
        )
        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.bin_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_center")
        self.bin_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bin")
        self.obstacle_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle"
        )
        self.upper_obstacle_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "upper_obstacle"
        )
        available_cubes = CUBE_NAMES[:1] if scene == "single" else CUBE_NAMES
        self.cube_joints = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in available_cubes]
        )
        self.cube_qpos = self.model.jnt_qposadr[self.cube_joints]
        self.cube_dof = self.model.jnt_dofadr[self.cube_joints]

        self.default_bin_pos = self.model.body_pos[self.bin_body].copy()
        self.default_obstacle_pos = (
            self.model.body_pos[self.obstacle_body].copy() if self.obstacle_body >= 0 else None
        )
        self.default_upper_pos = (
            self.model.body_pos[self.upper_obstacle_body].copy()
            if self.upper_obstacle_body >= 0
            else None
        )
        self.render_size = render_size
        self.renderer: mujoco.Renderer | None = None
        self.attached_cube: int | None = None
        self.goal_index = 0
        self.obstacle_zone = "center"
        self.reset()

    @property
    def state_dim(self) -> int:
        return 13 if self.scene == "single" else 19

    @property
    def action_dim(self) -> int:
        return 4

    def _reset_keyframe(self) -> None:
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "student_start"
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

    def reset(self) -> np.ndarray:
        self._reset_keyframe()
        self.attached_cube = None
        if self.scene == "single":
            self.data.qpos[self.cube_qpos[0] : self.cube_qpos[0] + 2] += self.rng.normal(
                0.0, 0.006, size=2
            )
            self.model.body_pos[self.obstacle_body] = self.default_obstacle_pos
            self.model.body_pos[self.upper_obstacle_body] = self.default_upper_pos
            if self.obstacle_mode == "train":
                self.obstacle_zone = "center"
                shift = self.rng.normal(0.0, OBSTACLE_POS_STD)
            else:
                draw = self.rng.random()
                if draw < ADVERSARIAL_CENTER_PROB:
                    self.obstacle_zone = "center"
                    offset, std = 0.0, OBSTACLE_POS_STD
                elif draw < ADVERSARIAL_CENTER_PROB + (1.0 - ADVERSARIAL_CENTER_PROB) / 2.0:
                    self.obstacle_zone = "+8 cm side"
                    offset, std = OBSTACLE_SHIFT_X, ADVERSARIAL_OBSTACLE_POS_STD
                else:
                    self.obstacle_zone = "-8 cm side"
                    offset, std = -OBSTACLE_SHIFT_X, ADVERSARIAL_OBSTACLE_POS_STD
                shift = offset + self.rng.normal(0.0, std)
            self.model.body_pos[self.obstacle_body, 0] += shift
            self.model.body_pos[self.upper_obstacle_body, 0] += shift
        else:
            slots = np.array([[-0.20, 0.35], [-0.20, 0.45], [-0.20, 0.55], [-0.20, 0.70]])
            assignment = self.rng.permutation(4)
            for cube_idx, slot_idx in enumerate(assignment[:3]):
                start = self.cube_qpos[cube_idx]
                self.data.qpos[start : start + 2] = slots[slot_idx] + self.rng.normal(
                    0.0, 0.005, size=2
                )
                self.data.qpos[start + 2] = 0.02
                self.data.qpos[start + 3 : start + 7] = (1.0, 0.0, 0.0, 0.0)
            self.model.body_pos[self.bin_body, :2] = slots[assignment[3]]
            self.goal_index = int(self.rng.integers(0, 3))

        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.ee_pos
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site])
        self.data.mocap_quat[0] = quat
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.joint_qpos]
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    @property
    def gripper(self) -> float:
        return float(self.data.qpos[self.joint_qpos[-1]])

    @property
    def bin_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_site].copy()

    @property
    def obstacle_pos(self) -> np.ndarray:
        if self.obstacle_body < 0:
            return np.zeros(3)
        return self.data.xpos[self.obstacle_body].copy()

    def cube_pos(self, index: int = 0) -> np.ndarray:
        start = self.cube_qpos[index]
        return self.data.qpos[start : start + 3].copy()

    @property
    def target_cube(self) -> int:
        return 0 if self.scene == "single" else self.goal_index

    def state(self) -> np.ndarray:
        if self.scene == "single":
            pieces = (
                self.ee_pos,
                np.array([self.gripper]),
                self.cube_pos(0),
                self.obstacle_pos,
                self.bin_pos,
            )
        else:
            goal = np.eye(3, dtype=np.float64)[self.goal_index]
            pieces = (
                self.ee_pos,
                np.array([self.gripper]),
                np.concatenate([self.cube_pos(i) for i in range(3)]),
                self.bin_pos,
                goal,
            )
        return np.concatenate(pieces).astype(np.float32)

    def _maybe_attach_or_release(self, gripper_target: float) -> None:
        if self.attached_cube is not None and gripper_target > 0.25:
            self.attached_cube = None
            return
        if self.attached_cube is not None or gripper_target >= 0.0:
            return
        cube_idx = self.target_cube
        grasp_point = self.ee_pos + np.array([0.0, 0.0, -0.065])
        # The coarse 20 Hz teaching controller and browser teleoperation can
        # contact the 4 cm cube before the jaw reaches its target.  A generous
        # 11 cm capture region keeps grasp acquisition reproducible; students
        # are told explicitly that this is the only assisted part of the task.
        if np.linalg.norm(self.cube_pos(cube_idx) - grasp_point) < 0.11:
            self.attached_cube = cube_idx

    def _update_attached_cube(self) -> None:
        if self.attached_cube is None:
            return
        cube_idx = self.attached_cube
        qpos_start = self.cube_qpos[cube_idx]
        dof_start = self.cube_dof[cube_idx]
        self.data.qpos[qpos_start : qpos_start + 3] = self.ee_pos + np.array(
            [0.0, 0.0, -0.065]
        )
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof_start : dof_start + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,):
            raise ValueError(f"expected action shape (4,), got {action.shape}")
        delta = np.clip(action[:3], -0.012, 0.012)
        gripper_target = float(np.clip(action[3], -0.17, 0.70))
        self.data.mocap_pos[0] += delta
        self.data.ctrl[self.actuator_ids[-1]] = gripper_target
        self._maybe_attach_or_release(gripper_target)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            self._update_attached_cube()
        return self.state()

    def success(self) -> bool:
        cube = self.cube_pos(self.target_cube)
        return bool(
            self.attached_cube is None
            and np.linalg.norm(cube[:2] - self.bin_pos[:2]) < 0.042
            and 0.0 < cube[2] < 0.075
        )

    def failed(self) -> bool:
        cube = self.cube_pos(self.target_cube)
        return bool(cube[2] < -0.02 or np.linalg.norm(cube[:2]) > 1.2)

    def render(self, camera: str = "angle") -> np.ndarray:
        if self.renderer is None:
            width, height = self.render_size
            self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def snapshot(self) -> Snapshot:
        return Snapshot(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
            mocap_pos=self.data.mocap_pos.copy(),
            mocap_quat=self.data.mocap_quat.copy(),
            time=float(self.data.time),
            attached_cube=self.attached_cube,
        )

    def restore(self, snapshot: Snapshot) -> None:
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.ctrl[:] = snapshot.ctrl
        self.data.mocap_pos[:] = snapshot.mocap_pos
        self.data.mocap_quat[:] = snapshot.mocap_quat
        self.data.time = snapshot.time
        self.attached_cube = snapshot.attached_cube
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


@dataclass(frozen=True)
class Waypoint:
    xyz: np.ndarray
    gripper: float
    dwell: int = 0


class ScriptedExpert:
    """Waypoint oracle that can start from an arbitrary policy-visited state."""

    def __init__(self, env: SO101Env) -> None:
        self.env = env
        self.index = 0
        self.dwell_steps = 0
        self.waypoints = self._build_plan()

    def _single_scene_route(self) -> tuple[float, float]:
        """Return a collision-aware side x coordinate and transit height."""
        obstacle_x = self.env.obstacle_pos[0]
        if self.env.obstacle_zone == "-8 cm side":
            # With the fixed end-effector orientation, raising to 18 cm makes
            # the lower arm strike the tall upper barrier in this zone. Pass
            # through the gap below it and leave extra clearance in x.
            return obstacle_x + 0.21, 0.11
        if self.env.obstacle_mode == "train":
            side = -1.0
        else:
            side = 1.0 if obstacle_x < -0.22 else -1.0
        return obstacle_x + side * 0.11, 0.18

    def _build_plan(self) -> list[Waypoint]:
        env = self.env
        cube = env.cube_pos(env.target_cube)
        bin_pos = env.bin_pos
        opened, closed = 0.70, -0.17
        if env.attached_cube is not None:
            return self._carry_plan(cube, bin_pos, closed, opened)

        safe_z, grasp_z = 0.18, 0.085
        if env.scene == "single":
            side_x, safe_z = self._single_scene_route()
        points: list[Waypoint] = [Waypoint(np.array([env.ee_pos[0], env.ee_pos[1], safe_z]), opened)]
        if env.scene == "single":
            points.extend(
                [
                    Waypoint(np.array([side_x, env.ee_pos[1], safe_z]), opened),
                    Waypoint(np.array([side_x, cube[1], safe_z]), opened),
                ]
            )
        points.extend(
            [
                Waypoint(np.array([cube[0], cube[1], 0.13]), opened),
                Waypoint(np.array([cube[0], cube[1], grasp_z]), opened),
                Waypoint(np.array([cube[0], cube[1], grasp_z]), closed, dwell=12),
                Waypoint(np.array([cube[0], cube[1], safe_z]), closed),
            ]
        )
        points.extend(self._carry_plan(cube, bin_pos, closed, opened, include_lift=False))
        return points

    def _carry_plan(
        self,
        cube: np.ndarray,
        bin_pos: np.ndarray,
        closed: float,
        opened: float,
        *,
        include_lift: bool = True,
    ) -> list[Waypoint]:
        env = self.env
        safe_z = 0.18
        points: list[Waypoint] = []
        if env.scene == "single":
            side_x, safe_z = self._single_scene_route()
        if include_lift:
            points.append(Waypoint(np.array([env.ee_pos[0], env.ee_pos[1], safe_z]), closed))
        if env.scene == "single":
            points.extend(
                [
                    Waypoint(np.array([side_x, cube[1], safe_z]), closed),
                    Waypoint(np.array([side_x, bin_pos[1], safe_z]), closed),
                ]
            )
        points.extend(
            [
                Waypoint(np.array([bin_pos[0], bin_pos[1], 0.14]), closed),
                Waypoint(np.array([bin_pos[0], bin_pos[1], 0.115]), closed),
                Waypoint(np.array([bin_pos[0], bin_pos[1], 0.115]), opened, dwell=14),
                Waypoint(np.array([bin_pos[0], bin_pos[1], 0.18]), opened, dwell=4),
            ]
        )
        return points

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def action(self) -> np.ndarray:
        if self.done:
            return np.array([0.0, 0.0, 0.0, 0.70], dtype=np.float32)
        waypoint = self.waypoints[self.index]
        error = waypoint.xyz - self.env.ee_pos
        close_enough = np.linalg.norm(error) < 0.012
        if close_enough:
            if self.dwell_steps < waypoint.dwell:
                self.dwell_steps += 1
            else:
                self.index += 1
                self.dwell_steps = 0
                return self.action()
        delta = np.clip(error, -0.012, 0.012)
        return np.r_[delta, waypoint.gripper].astype(np.float32)


def run_expert_episode(
    env: SO101Env, *, max_steps: int = 420
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Run one expert episode and return time-aligned states and actions."""
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    expert = ScriptedExpert(env)
    for _ in range(max_steps):
        states.append(env.state())
        action = expert.action()
        actions.append(action)
        env.step(action)
        if env.success() or env.failed():
            break
    return np.stack(states), np.stack(actions), env.success()


def chunk_episode(
    states: np.ndarray, actions: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create full state-to-action-chunk examples without crossing an episode."""
    count = len(states) - chunk_size + 1
    if count <= 0:
        return states[:0], np.empty((0, chunk_size, actions.shape[-1]), dtype=np.float32)
    chunks = np.stack([actions[t : t + chunk_size] for t in range(count)])
    return states[:count].astype(np.float32), chunks.astype(np.float32)


def generate_expert_dataset(
    env_factory: Callable[[int], SO101Env],
    *,
    episodes: int,
    chunk_size: int,
    seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    """Generate successful demonstrations and preserve episode identities."""
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    successes = 0
    for episode in range(episodes):
        env = env_factory(seed_offset + episode)
        states, actions, success = run_expert_episode(env)
        chunk_states, chunks = chunk_episode(states, actions, chunk_size)
        state_parts.append(chunk_states)
        action_parts.append(chunks)
        episode_parts.append(np.full(len(chunk_states), episode, dtype=np.int64))
        successes += int(success)
        env.close()
    if successes != episodes:
        raise RuntimeError(f"scripted expert succeeded in {successes}/{episodes} episodes")
    return {
        "state": np.concatenate(state_parts),
        "action_chunk": np.concatenate(action_parts),
        "episode": np.concatenate(episode_parts),
    }


def expert_chunk(env: SO101Env, chunk_size: int) -> np.ndarray:
    """Label the current policy-visited state without changing the rollout."""
    snapshot = env.snapshot()
    expert = ScriptedExpert(env)
    actions = []
    for _ in range(chunk_size):
        action = expert.action()
        actions.append(action)
        env.step(action)
    env.restore(snapshot)
    return np.stack(actions).astype(np.float32)


def run_chunked_policy(
    env: SO101Env,
    predict_chunk: Callable[[np.ndarray], np.ndarray],
    *,
    max_steps: int = 420,
    visit_callback: Callable[[np.ndarray, np.ndarray], None] | None = None,
    capture_frames: bool = True,
) -> tuple[bool, list[np.ndarray]]:
    """Execute complete open-loop chunks and optionally label visited states."""
    frames: list[np.ndarray] = [env.render()] if capture_frames else []
    steps = 0
    while steps < max_steps and not env.success() and not env.failed():
        state = env.state()
        policy_chunk = np.asarray(predict_chunk(state), dtype=np.float32)
        if visit_callback is not None:
            visit_callback(state.copy(), expert_chunk(env, len(policy_chunk)))
        for action in policy_chunk:
            env.step(action)
            steps += 1
            if capture_frames and steps % 20 == 0:
                frames.append(env.render())
            if env.success() or env.failed() or steps >= max_steps:
                break
    if capture_frames:
        frames.append(env.render())
    return env.success(), frames


def evaluate_policy(
    env_factory: Callable[[int], SO101Env],
    predict_chunk: Callable[[np.ndarray], np.ndarray],
    seeds: Iterable[int],
) -> np.ndarray:
    outcomes = []
    for seed in seeds:
        env = env_factory(int(seed))
        success, _ = run_chunked_policy(env, predict_chunk, capture_frames=False)
        outcomes.append(success)
        env.close()
    return np.asarray(outcomes, dtype=bool)
