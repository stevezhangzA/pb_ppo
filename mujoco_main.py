import math
import sys
import gymnasium as gym
import torch
import warnings
from stable_baselines3 import PPO
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Tuple
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

import numpy as np
import torch as th
from torch.nn import functional as F
import pickle
import logging

import time

import wandb
import argparse 

import pickle

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")
SelfPPO = TypeVar("SelfPPO", bound="PPO")

device='cuda'

class custom_ppo(PPO):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {"MlpPolicy": ActorCriticPolicy,
                                                             "CnnPolicy": ActorCriticCnnPolicy,
                                                             "MultiInputPolicy": MultiInputActorCriticPolicy,}
    def __init__(self,policy: Union[str, Type[ActorCriticPolicy]],
                     env: Union[GymEnv, str],
                     learning_rate: Union[float, Schedule] = 3e-4,
                     n_steps: int = 2048,
                     batch_size: int = 64,
                     n_epochs: int = 10,
                     gamma: float = 0.99,
                     gae_lambda: float = 0.95,
                     clip_range: Union[float, Schedule] = 0.2,
                     clip_range_vf: Union[None, float, Schedule] = None,
                     normalize_advantage: bool = True,
                     ent_coef: float = 0.0,
                     vf_coef: float = 0.5,
                     max_grad_norm: float = 0.5,
                     use_sde: bool = False,
                     sde_sample_freq: int = -1,
                     target_kl: Optional[float] = None,
                     stats_window_size: int = 100,
                     tensorboard_log: Optional[str] = None,
                     policy_kwargs: Optional[Dict[str, Any]] = None,
                     verbose: int = 0,
                     seed: Optional[int] = 2,
                     device: Union[th.device, str] = "auto",
                     _init_setup_model: bool = True,
                     supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,):
        super().__init__(policy,
                         env,
                         learning_rate=learning_rate,
                         n_steps=n_steps,
                         gamma=gamma,
                         gae_lambda=gae_lambda,
                         ent_coef=ent_coef,
                         vf_coef=vf_coef,
                         max_grad_norm=max_grad_norm,
                         use_sde=use_sde,
                         sde_sample_freq=sde_sample_freq,
                         stats_window_size=stats_window_size,
                         tensorboard_log=tensorboard_log,
                         policy_kwargs=policy_kwargs,
                         verbose=verbose,
                         device=device,
                         seed=seed,
                         _init_setup_model=False)
        
        self.supported_action_spaces=(spaces.Box,
                                      spaces.Discrete,
                                      spaces.MultiDiscrete,
                                      spaces.MultiBinary,)
        
        self.bandit=None
        self.arm_visitations=None
        self.arm_expectations=None
        self.total_return=0
        self.N=0
        self.C=5
        self.discount=0.99
        self.use_advantage=False
        self.average_times=2
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(f"You have specified a mini-batch size of {batch_size},"
                              f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                              f" after every {untruncated_batches} untruncated mini-batches,"
                              f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                              f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                              f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})")
                
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        if _init_setup_model:
            self._setup_model()
        self.use_wandb=True
        self.global_step=0
        self.use_bandit=False
        self.fixed_bandit=0.05
        
    def times_update(self,arm_index):
        self.N+=1
        self.arm_visitations[arm_index]+=1

    def expectation_update(self,arm_index,return_):
        self.arm_expectations[arm_index]=self.discount*self.arm_expectations[arm_index]+return_

    def ucb_computing(self):
        max_return=max(self.arm_expectations)
        if self.use_advantage:
            normalized_expectation=[i/(max_return+0.001)-self.total_return for i in self.arm_expectations]
        else:
            normalized_expectation=[i/(max_return+0.001) for i in self.arm_expectations]
        ucb_list = [normalized_expectation[id_] + self.C * math.sqrt(self.N / (self.arm_visitations[id_] + 0.001)) for id_ in range(len(self.bandit))]
        index_=np.argmax(np.array(ucb_list))
        return index_,self.bandit[index_]

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True
        # train for n_epochs epochs
        if self.use_bandit:
            index_, clip_range = self.ucb_computing() # sample a bandit arm for ppo
            self.times_update(index_)
        ff = open('RATIO.txt','wb')
        kkll = open('KKLL.txt','wb')
        approx_kl_divs = []
        RATIO = []
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            # sample arm
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                ratio1 = (log_prob * th.log(log_prob / rollout_data.old_log_prob)).mean()
                RATIO.append(ratio1.item())
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp( values - rollout_data.old_values, -clip_range_vf, clip_range_vf )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    # entropy loss 
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob 
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy() # kl divergence
                    approx_kl_divs.append(approx_kl_div)
                # logging the kl divergence between new and old policies
                if self.use_wandb:
                    wandb.log({'kl_divergence':approx_kl_div,
                               'epoch':self.global_step})
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                # Optimization step
                self.policy.optimizer.zero_grad() 
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            # update arm
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        pickle.dump(np.mean(RATIO),ff)
        pickle.dump(np.mean(approx_kl_divs),kkll)
        self.logger.record("train/RATIO", ratio1.item())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        #wandb.log( {'train/loss':  loss.item(), 
        #            'epoch': self.global_step,})
        if self.use_bandit:
            total_return=0
            # model.average_times=cfg.average_times
            # model.use_advantage=cfg.use_advantage
            for eval_times in range(self.average_times):
                cur_return=0
                with torch.no_grad():
                    obs = self.env.reset()
                    for i in range(1000):
                        action, _states = self.predict(obs, deterministic=True)
                        obs, reward, done, info = self.env.step(action)
                        cur_return+=reward
                total_return+=cur_return
            total_return/=self.average_times
            self.expectation_update(index_,total_return)
            self.total_return=total_return+self.total_return*self.discount

    def learn_rewrite(self: SelfOnPolicyAlgorithm,
                    total_timesteps: int,
                    callback: MaybeCallback = None,
                    log_interval: int = 1,
                    tb_log_name: str = "OnPolicyAlgorithm",
                    reset_num_timesteps: bool = True,
                    progress_bar: bool = False) -> SelfOnPolicyAlgorithm:
        iteration = 0
        total_timesteps, callback = self._setup_learn(total_timesteps,
                                                    callback,
                                                    reset_num_timesteps,
                                                    tb_log_name,
                                                    progress_bar,)
        callback.on_training_start(locals(), globals())
        assert self.env is not None
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            if continue_training is False:
                break
            iteration += 1
            self.global_step=iteration
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                if self.use_wandb:
                    wandb.log({"rollout/ep_rew_mean": 
                               safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                               'epoch':iteration})
                    wandb.log({"rollout/ep_len_mean": 
                               safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                               "epoch":iteration})
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
            self.train()
        callback.on_training_end()
        return self

    def learn(self: SelfPPO,
             total_timesteps: int,
             callback: MaybeCallback = None,
             log_interval: int = 1,
             tb_log_name: str = "PPO",
             reset_num_timesteps: bool = True,
             progress_bar: bool = False,) -> SelfPPO:
        return self.learn_rewrite( total_timesteps=total_timesteps,
                              callback=callback,
                              log_interval=log_interval,
                              tb_log_name=tb_log_name,
                              reset_num_timesteps=reset_num_timesteps,
                              progress_bar=progress_bar )

# intialize the configuration
cfg_raw=argparse.ArgumentParser()
cfg_raw.add_argument('--seed',type=int,default=0)
cfg_raw.add_argument('--logging_path',type=str,default='human_preference')
cfg_raw.add_argument('--env_name',type=str,default='Hopper-v3')
cfg_raw.add_argument('--num_arms',type=int,default=6)
cfg_raw.add_argument('--min_threeshold',type=float,default=0.02)
cfg_raw.add_argument('--max_threeshold',type=float,default=0.5)
cfg_raw.add_argument('--proj_name' ,type=str ,default='')
cfg_raw.add_argument('--name' ,type=str ,default='')
cfg_raw.add_argument('--eval_freq',type=int,default=500)
cfg_raw.add_argument('--total_training_steps',type=int,default=20_00000)
cfg_raw.add_argument('--use_wandb',default=True,type=bool)
cfg_raw.add_argument('--verbose',default=1,type=int)
cfg_raw.add_argument('--use_advantage',default=0,type=int)
cfg_raw.add_argument('--average_times',default=2,type=int)
cfg_raw.add_argument('--use_',default=0,type=int)
cfg_raw.add_argument('--clip_range',default=0.2,type=float,help='fixed surrogate trust region')
cfg=cfg_raw.parse_args()

import gym
eval_env = gym.make(cfg.env_name,render_mode='rgb_array')
env=gym.make(cfg.env_name,render_mode='rgb_array')
dict_args=vars(cfg)

if cfg.use_==0:
    cfg.use_bandit=False
else:
    cfg.use_bandit=True
if cfg.use_advantage==0:
    cfg.use_advantage=False
else:
    cfg.use_advantage=True

wandb_config={}
wandb_config={k: dict_args[k] for k in dict_args}
cfg.name=f'{cfg.env_name}_num_bandits_{cfg.num_arms}_minthree_{cfg.min_threeshold}_max_threeshold_{cfg.max_threeshold}_evalfreq_{cfg.eval_freq}_total_traing_steps_{cfg.total_training_steps}_seed_{cfg.seed}_bandit_advantage_{cfg.use_advantage}_defaultcliprange_{cfg.clip_range}_use_bandit_{cfg.use_bandit}'
wandb.init(project=cfg.proj_name,
           name=cfg.name,
           config=dict_args,
           resume=cfg.seed)

if cfg.use_advantage:
    folder_name='wi_ad'
else:
    folder_name='wo_ad'
if not cfg.use_bandit:
    eval_callback = EvalCallback(eval_env,
                                best_model_save_path=f'./{cfg.env_name}/{cfg.seed}',
                                log_path=f'./humanpref_{cfg.logging_path}_{cfg.clip_range}_{cfg.num_arms}/{folder_name}/{cfg.seed}', 
                                eval_freq=cfg.eval_freq,
                                deterministic=True, render=False)
else:
    eval_callback = EvalCallback(eval_env,
                                best_model_save_path=f'./{cfg.env_name}/{cfg.seed}',
                                log_path=f'./{cfg.logging_path}_maxthree_{cfg.max_threeshold}_min_three_{cfg.min_threeshold}_{cfg.num_arms}/{folder_name}/{cfg.seed}', 
                                eval_freq=cfg.eval_freq,
                                deterministic=True, render=False)

model = custom_ppo("MlpPolicy", 
                   env, 
                   verbose=cfg.verbose,
                   seed=cfg.seed,
                   clip_range=cfg.clip_range,)
model.average_times=cfg.average_times
model.use_advantage=cfg.use_advantage
model.init_wandb=True

# all bandits
initialized_bandit=list(np.linspace(cfg.min_threeshold,
                                    cfg.max_threeshold,
                                    cfg.num_arms))

# initialize the fixed surrogate trust region
model.use_bandit=cfg.use_bandit
logging.info(f'use bandit : {cfg.use_bandit}')
model.bandit=initialized_bandit
model.arm_visitations=[0 for i in initialized_bandit]
model.arm_expectations=[0 for i in initialized_bandit]        
# training model
model.learn(total_timesteps=cfg.total_training_steps,
            callback=eval_callback)



