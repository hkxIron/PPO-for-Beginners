"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""
from typing import *

import gym
import sys
import torch

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
from gym import Env
from torch import nn

def train(env:Env, hyperparameters:Dict[str, Any], actor_model_file, critic_model_file):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model_file != '' and critic_model_file != '':
		print(f"Loading in {actor_model_file} and {critic_model_file}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model_file))
		model.critic.load_state_dict(torch.load(critic_model_file))
		print(f"Successfully loaded.", flush=True)
	elif actor_model_file != '' or critic_model_file != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=200_000_000)

def test(env:Env, actor_model_file:nn.Module):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model_file - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model_file}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model_file == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	observation_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNN(observation_dim, action_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model_file))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True)

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

	"""
	Pendulum-v1
	状态：cos(theta), sin(theta) , thetadot（角速度）, 是一个3维向量
	theta:杆子的角度， 竖直为0度，顺时针方向来计算角度，所以角度范围是[0,360],例如图中大概是140度的样子
	
	动作：往左转还是往右转，用力矩来衡量，即力乘以力臂,维度为1,是一个标量。范围[-2,2]：
	env.action_space:
	Box([-2], [-2,], (1,), float32)
	
	奖励：
	def angle_normalize(x):
		return ((x+np.pi)%(2*np.pi)) - np.pi	
		
	costs = angle_normalize(th)**2 + 0.1*thdot**2 + 0.001*(u**2)
	th就是上面的角度，thdot就是角速度，u就是你输入的动作值[-2,2]。
	总的来说，单摆越直立拿到的奖励越高，越偏离，奖励越低。
	其目标是：从任意状态出发，施加一系列的力矩，使得摆杆可以坚直向上
	
	游戏结束：
		200步后游戏结束。所以要在200步内拿到的分越高越好。
	
	env.observation_space#查看状态空间
	env.action_space#查看动作空间
	"""
	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
	env = gym.make('Pendulum-v1', render_mode="rgb_array")

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model_file=args.actor_model, critic_model_file=args.critic_model)
	else:
		test(env=env, actor_model_file=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
