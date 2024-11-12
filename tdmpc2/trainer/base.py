from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper
from ..tdmpc2 import TDMPC2

class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env: PixelWrapper | MultitaskWrapper | TensorWrapper, agent: TDMPC2, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		print('Architecture:', self.agent.model)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
