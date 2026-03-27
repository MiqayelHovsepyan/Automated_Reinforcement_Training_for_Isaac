# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy with a side-view camera for visual gait inspection.

This is a modified version of scripts/rsl_rl/play.py optimized for auto-train
visual inspection. Key differences:
- Side-view camera at robot height following a single robot (not overhead of many)
- Defaults to 4 environments (not 50) for clear individual robot visibility
- Configurable camera distance, height, and azimuth via CLI args

Usage:
    python .claude/skills/auto_train/resources/play_for_inspection.py \
        --task=Isaac-WTW-Flat-Ayg-Play-v0 \
        --checkpoint=logs/rsl_rl/.../model_500.pt \
        --video --video_length=300 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports — cli_args lives in scripts/rsl_rl/, compute path from cf_lab root
import os

# Script lives at .claude/skills/auto_train/resources/ — go up 4 levels to cf_lab root
_cf_lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_cf_lab_dir, "scripts", "rsl_rl"))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy with side-view camera for visual inspection.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments (default: 4 for clear view).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# Camera configuration
parser.add_argument(
    "--camera-distance", type=float, default=2.0, help="Camera distance from robot (meters). Default: 2.0"
)
parser.add_argument(
    "--camera-height", type=float, default=0.4, help="Camera height above ground (meters). Default: 0.4"
)
parser.add_argument(
    "--camera-azimuth", type=float, default=90.0,
    help="Camera azimuth angle in degrees. 0=front, 90=side, 180=back. Default: 90 (side view)"
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    ViewerCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import cf_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using side-view camera for gait inspection."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # Force small number of envs for clear visibility
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── Override camera for side-view inspection ──
    # Convert azimuth to camera position relative to robot
    azimuth_rad = math.radians(args_cli.camera_azimuth)
    cam_x = args_cli.camera_distance * math.cos(azimuth_rad)
    cam_y = -args_cli.camera_distance * math.sin(azimuth_rad)
    cam_z = args_cli.camera_height

    env_cfg.viewer = ViewerCfg(
        eye=(cam_x, cam_y, cam_z),
        lookat=(0.0, 0.0, 0.25),  # Robot center of mass height
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )
    print(f"[INSPECT] Camera: distance={args_cli.camera_distance}m, height={args_cli.camera_height}m, "
          f"azimuth={args_cli.camera_azimuth}°")
    print(f"[INSPECT] Camera eye=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), "
          f"lookat=(0, 0, 0.25), origin=asset_root, following robot 0")
    print(f"[INSPECT] Environments: {args_cli.num_envs} (reduced for clear individual visibility)")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording side-view inspection video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation (skip in headless/video mode for speed)
        if not args_cli.video:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
