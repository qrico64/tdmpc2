from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


# cfg.eval_episodes: Number of episodes to plot for evaluation.
# cfg.save_video: Whether or not to save the videos of the evaluations.
class DaggerTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, *kwargs)
  
  def eval(self):
    """Evaluate a TD-MPC2 agent."""
    ep_rewards, ep_successes = [], []

    # set mpc (planning) to false. This is so that we don't plan during self.agent.act()
    original_mpc = self.agent.cfg.mpc
    self.agent.cfg.mpc = False

    # copied code from OnlineTrainer.
    for i in range(self.cfg.eval_episodes):
      obs, done, ep_reward, t = self.env.reset(), False, 0, 0
      if self.cfg.save_video:
        self.logger.video.init(self.env, enabled=(i==0))
      while not done:
        torch.compiler.cudagraph_mark_step_begin()
        action = self.agent.act(obs, t0=t==0, eval_mode=True)
        obs, reward, done, info = self.env.step(action)
        ep_reward += reward
        t += 1
        if self.cfg.save_video:
          self.logger.video.record(self.env)
      ep_rewards.append(ep_reward)
      ep_successes.append(info['success'])
      if self.cfg.save_video:
        self.logger.video.save(self._step)
    
    # set mpc (planning) back to original value.
    self.agent.cfg.mpc = original_mpc

    # return average reward and success rate.
    return dict(
      episode_reward=np.nanmean(ep_rewards),
      episode_success=np.nanmean(ep_successes),
    )

  def train(self):
    """Train a TD-MPC2 agent."""
    raise NotImplementedError
