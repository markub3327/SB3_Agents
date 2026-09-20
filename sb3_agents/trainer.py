import argparse
import ale_py
import gymnasium
import stable_retro as retro
import minigrid
from minigrid.wrappers import FlatObsWrapper
from gymnasium.wrappers import TimeLimit, FlattenObservation
from schedule import CosineAnnealingLR
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    WarpFrame,
)
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)
from utils import load_hyperparams
from wandb.integration.sb3 import WandbCallback
import wandb
import torch


# Register environments
gymnasium.register_envs(ale_py)


def make_retro_env(env_name):
    def _body():
        """
        Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
        """
        env = retro.make(env_name, retro.State.DEFAULT, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=8192)
        env = Monitor(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env, width=96, height=96)
        env = ClipRewardEnv(env)
        return env

    return _body

def make_minigrid_env(env_name):
    def _body():
        env = gymnasium.make(env_name, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        return env

    return _body

def make_toy_text_env(env_name):
    def _body():
        env = gymnasium.make(env_name, render_mode="rgb_array")
        env = FlattenObservation(env)
        return env

    return _body


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning project training PPO agents on Atari, Retro (Sonic, Mario) or classic environments using Stable Baselines 3, featuring dataset generation and WanDB experiment tracking."
    )
    parser.add_argument(
        "--emulator",
        type=str,
        required=True,
        choices=["ale", "retro", "classic", "minigrid", "toy"],
        help="The name of the emulator ['ale', 'retro', 'classic', 'minigrid', 'toy']",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="The name of the environment (e.g., 'BreakoutNoFrameskip-v4', 'SuperMarioBros3-Nes-v0', 'LunarLander-v3')",
    )
    args = parser.parse_args()

    # For Atari console
    if "ale" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.emulator, "./sb3_agents/hyperparams.yml")
        # Create train environment
        train_env = make_atari_env(
            args.env,
            n_envs=config["n_envs"],
            wrapper_kwargs={"terminal_on_life_loss": True},
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        train_env = VecFrameStack(train_env, n_stack=config["frame_stack"])
        train_env = VecTransposeImage(train_env)
        # Create eval environment
        eval_env = make_atari_env(
            args.env,
            n_envs=config["n_envs"],
            wrapper_kwargs={"terminal_on_life_loss": False},
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        eval_env = VecFrameStack(eval_env, n_stack=config["frame_stack"])
        eval_env = VecTransposeImage(eval_env)
    # For stable-retro consoles
    elif "retro" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.emulator, "./sb3_agents/hyperparams.yml")
        # Create train environment
        train_env = make_vec_env(
            make_retro_env(args.env),
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        train_env = VecFrameStack(train_env, n_stack=config["frame_stack"])
        train_env = VecTransposeImage(train_env)
        # Create eval environment
        eval_env = make_vec_env(
            make_retro_env(args.env),
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        eval_env = VecFrameStack(eval_env, n_stack=config["frame_stack"])
        eval_env = VecTransposeImage(eval_env)
    # For Classic
    elif "classic" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.env, "./sb3_agents/hyperparams.yml")
        # Create train environment
        train_env = make_vec_env(
            args.env,
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        # Create eval environment
        eval_env = make_vec_env(
            args.env,
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        if config["policy"] == "CnnPolicy":
            train_env = VecTransposeImage(train_env)
            eval_env = VecTransposeImage(eval_env)
    elif "minigrid" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.env, "./sb3_agents/hyperparams.yml")
        # Create train environment
        train_env = make_vec_env(
            make_minigrid_env(args.env),
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        # Create eval environment
        eval_env = make_vec_env(
            make_minigrid_env(args.env),
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
    elif "toy" in args.emulator.lower():
        # Load PPO configuration
        config = load_hyperparams(args.env, "./sb3_agents/hyperparams.yml")
        # Create train environment
        train_env = make_vec_env(
            make_toy_text_env(args.env),
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
        # Create eval environment
        eval_env = make_vec_env(
            make_toy_text_env(args.env),
            n_envs=config["n_envs"],
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )

    # Use normalization
    if config["normalize"]:
        train_env = VecNormalize(
            train_env,
            training=True,
            norm_obs=config["normalize"]["norm_obs"],
            norm_reward=config["normalize"]["norm_reward"],
            gamma=config["gamma"],
        )
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=config["normalize"]["norm_obs"],
            norm_reward=config["normalize"]["norm_reward"],
            gamma=config["gamma"],
        )

    # Initialize WanDB
    run = wandb.init(
        project="ppo-sb3",
        config={
            "policy_type": config["policy"],
            "total_timesteps": config["n_timesteps"],
            "env_name": args.env,
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,
    )

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=max(config["eval_freq"] // config["n_envs"], 1),
        best_model_save_path=f"./save/{args.env}",
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Define policy keyword arguments
    policy_kwargs = {
        'optimizer_class': torch.optim.AdamW if config['policy_kwargs']['optimizer_class'].lower() == 'adamw' else torch.optim.Adam,
        'optimizer_kwargs': config['policy_kwargs']['optimizer_kwargs'],
        'ortho_init': config['policy_kwargs']['ortho_init'],
        'activation_fn': torch.nn.ReLU if config['policy_kwargs']['activation_fn'].lower() == 'relu' else torch.nn.Tanh
    }
    
    # Create the PPO model
    model = PPO(
        policy=config["policy"],
        env=train_env,
        n_steps=config["n_steps"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        learning_rate=CosineAnnealingLR(config["learning_rate"]),
        clip_range=config["clip_range"],
        vf_coef=config["vf_coef"],
        ent_coef=config["ent_coef"],
        normalize_advantage=config["normalize_advantage"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=f"./logs/{run.id}",
    )

    # Train the model
    model.learn(
        total_timesteps=config["n_timesteps"],
        callback=[eval_callback, WandbCallback(verbose=1)],
        progress_bar=True,
    )

    # Finish the WanDB run
    run.finish()
