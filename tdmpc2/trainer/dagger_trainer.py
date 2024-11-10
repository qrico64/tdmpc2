from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class DaggerTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, *kwargs)
  
  def eval(self):
    """Evaluate a TD-MPC2 agent."""
    raise NotImplementedError

  def train(self):
    """Train a TD-MPC2 agent."""
    raise NotImplementedError
