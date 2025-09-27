# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

from drqv2 import DrQV2AgentOni, Critic, CriticRBN, CriticCBN, CriticBN
from drq import DrQAgent
from datetime import datetime

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        if cfg.agent._target_ == 'drq.DrQAgent':
            cfg.agent.obs_shape = self.train_env.observation_spec().shape
            cfg.agent.action_shape = self.train_env.action_spec().shape
            cfg.agent.action_range = [
                float(self.train_env.action_spec().minimum),
                float(self.train_env.action_spec().maximum)
            ]


        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.logger.writer.info(f'workspace: {self.work_dir}')
        self.logger.writer.info(cfg)
        self.logger.writer.info(self.agent)
        if(isinstance(self.agent, DrQAgent)):
            self.logger.writer.info(self.agent.actor.encoder)
        else:
            self.logger.writer.info(self.agent.encoder)
        self.logger.writer.info(self.agent.actor)
        self.logger.writer.info(self.agent.critic)
        if(isinstance(self.agent, DrQV2AgentOni)):
            self.logger.writer.info('encoder optimizer: ')
            self.logger.writer.info(self.agent.encoder_opt)
        self.logger.writer.info('actor optimizer: ')
        self.logger.writer.info(self.agent.actor_opt)
        self.logger.writer.info('critic optimizer: ')
        self.logger.writer.info(self.agent.critic_opt)
        self.logger.writer.info('update modes: ' + self.agent.update_mode)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.critic_bn_layer = 2

    def setup(self):
        # create logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.work_dir = Path(f'./runs/{timestamp}')
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def get_data(self, test_episode=20):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(test_episode)
        state_li = []
        action_li = []
        reward_li = []
        mc_return_li = []
        discount = self.cfg.discount #0.99

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            state = time_step.observation
            episode_rewards = []
            episode_states = []
            episode_actions = []

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                next_state = time_step.observation

                total_reward += time_step.reward
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(time_step.reward)
                state = next_state
                step += 1

            mc_returns = np.zeros(len(episode_rewards))
            mc_return = 0
            for i in range(len(episode_rewards) - 1, -1, -1):
                mc_return = episode_rewards[i] + discount * mc_return
                mc_returns[i] = mc_return

            state_li.extend(episode_states)
            action_li.extend(episode_actions)
            reward_li.extend(episode_rewards)
            mc_return_li.extend(mc_returns)

            episode += 1

        ori_obs = torch.Tensor(np.array(state_li)).to(self.device)
        ori_action = torch.Tensor(np.array(action_li)).to(self.device)
        reward = torch.Tensor(np.array(reward_li)).to(self.device).unsqueeze(1)
        mc_return = torch.Tensor(np.array(mc_return_li)).to(self.device).unsqueeze(1)

        data = {}
        data["ori_obs"] = ori_obs
        data["ori_action"] = ori_action
        data["reward"] = reward
        data["mc_return"] = mc_return
        return data

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        if len(self.episode_bias_mean) > 0:
            est_bias_mean = np.mean(self.episode_bias_mean)
            est_bias_std = np.mean(self.episode_bias_std)
            self.episode_bias_mean = []
            self.episode_bias_std = []
            actor_grad_sim_mean = np.mean(self.episode_actor_grad_sim_mean)
            actor_grad_sim_std = np.mean(self.episode_actor_grad_sim_std)
            actor_grad_norm = np.mean(self.episode_actor_grad_norm)
            self.episode_actor_grad_sim_mean = []
            self.episode_actor_grad_sim_std = []
            self.episode_actor_grad_norm = []
        else:
            est_bias_mean = 0.0
            est_bias_std = 0.0
            actor_grad_sim_mean = 0.0
            actor_grad_sim_std = 0.0
            actor_grad_norm = 0.0
                
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('est_bias_mean', est_bias_mean)
            log('est_bias_std', est_bias_std)
            log('actor_grad_sim_mean', actor_grad_sim_mean)
            log('actor_grad_sim_std', actor_grad_sim_std)
            log('actor_grad_norm', actor_grad_norm)

    def train(self):
        self.episode_actor_grad_sim_mean = []
        self.episode_actor_grad_sim_std = []
        self.episode_actor_grad_norm = []
        self.episode_bias_mean = []
        self.episode_bias_std = []
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if self.global_step % 1000 == 0:
                    data = self.get_data()
                else:
                    data = None
                if data is not None:
                    metrics = self.agent.update(self.replay_iter, self.global_step, data)
                    actor_grad_sim_mean, actor_grad_sim_std, actor_grad_norm = self.agent.actor_grad_rec.sim_cal()
                    self.episode_actor_grad_sim_mean.append(actor_grad_sim_mean)
                    self.episode_actor_grad_sim_std.append(actor_grad_sim_std)
                    self.episode_actor_grad_norm.append(actor_grad_norm)
                    self.episode_bias_mean.append(self.agent.est_bias_mean)
                    self.episode_bias_std.append(self.agent.est_bias_std)
                else:
                    metrics = self.agent.update(self.replay_iter, self.global_step, data=None)
                    
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config_oni', version_base="1.1")
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()