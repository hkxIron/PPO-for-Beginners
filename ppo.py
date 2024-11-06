from typing import *

import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from gym import Env

from network import FeedForwardNN

"""
The file contains the PPO class to train with.
NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
		It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg


"""
class PPO:
	"""
		PPO 是on-policy的策略

		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class:Type[FeedForwardNN], env:Env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# # PPO 初始化⽤于训练的超参数
		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# 提取环境信息
		# Extract environment information
		self.env = env
		self.observation_dim = env.observation_space.shape[0] # 在单摆中的状态空间，obs_dim=3
		self.action_dim = env.action_space.shape[0] # 单摆中的动作空间，act_dim=1

		# 初始化协⽅差矩阵，⽤于查询actor⽹络的action
		# Initialize actor and critic networks
		self.actor:FeedForwardNN = policy_class(in_dim=self.observation_dim, out_dim=self.action_dim)                                                   # ALG STEP 1
		self.critic:FeedForwardNN = policy_class(in_dim=self.observation_dim, out_dim=1)

		# Initialize optimizers for actor and critic
		# 为演员和评论家初始化优化器
		self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		# 初始化协方差矩阵，用于从中采样actor网络的action
		self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# 这个记录器将帮助我们打印出每个迭代的摘要
		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_time': time.time_ns(),
			't_so_far': 0,          # timesteps so far,  到目前为止的时间步数
			'i_so_far': 0,          # iterations so far, 到目前为止的迭代次数
			'batch_lens': [],       # episodic lengths in batch, 批次中的episodic长度
			'batch_rews': [],       # episodic returns in batch,
			'actor_losses': [],     # losses of actor network in current iteration
			'critic_losses': [],  # losses of actor network in current iteration
		}

	def learn(self, total_timesteps:int=200_000_000):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		timestep_so_far = 0 # Timesteps simulated so far, 到⽬前为⽌仿真的时间步数
		iter_so_far = 0 # Iterations ran so far, 到⽬前为⽌，已运⾏的迭代次数

		while timestep_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)

			# batch_observation_k:[number of timesteps, dimension of observation]
			# batch_actions_k: [number of timesteps, dimension of action]
			# batch_log_probs_k: [number of timesteps]
			# batch_discount_reward: [number of timesteps]
			# batch_lens_k:[number of episodes]
			# # 利用第k次模型收集批量实验数据, on-policy
			batch_observation_k, batch_actions_k, batch_log_probs_k, batch_discount_reward_k, batch_lens_k = self.get_actor_play_data()                     # ALG STEP 3

			# 计算收集这一批数据的时间步数
			# Calculate how many timesteps we collected this batch
			timestep_so_far += np.sum(batch_lens_k) # 这个batch中下了多少步棋

			# Increment the number of iterations
			iter_so_far += 1

			# 记录到目前为止的时间步数和到目前为止的迭代次数
			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = timestep_so_far
			self.logger['i_so_far'] = iter_so_far

			# 计算第k次迭代的state的优势函数
			# Calculate advantage at k-th iteration
			# batch_observation_k:[timesteps, state_dim=3]
			# batch_actions_k: [timesteps, action_dim=3]
			# state_value_k:[timesteps], 由critic value function计算而来
			state_value_k, _ = self.evaluate(batch_observation_k, batch_actions_k) # 每个时间步的状态值

			# 计算第k次迭代的advantage
			# 优势函数： Advantage(t) = Rt + gamma* V(t+1) - V(t)
			#
			# batch_discount_reward: [timesteps]
			# advantage_k: [timesteps]
			advantage_k = batch_discount_reward_k - state_value_k.detach()                                                                       # ALG STEP 5

			# 将优势归一化 在理论上不是必须的，但在实践中，它减少了我们优势的方差，使收敛更加稳定和快速。
			# 添加这个是因为在没有这个的情况下，解决一些环境的问题太不稳定了。
			# One of the only tricks I use that isn't in the pseudocode.
			# Normalizing advantages isn't theoretically necessary,
			#
			# but in practice it decreases the variance of
			# our advantages and makes convergence much more stable and faster.
			# I added this because solving some environments was too unstable without it.
			advantage_k = (advantage_k - advantage_k.mean()) / (advantage_k.std() + 1e-10) # 用均值方差归一化是额外添加的,会更稳定，类似于ak大神的做法

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration): # ALG STEP 6 & 7, 每次迭代更新模型10次
				# critic 对当前状态评价10次,并同时更新 actor, critic模型
				# Calculate V_phi and pi_theta(a_t | s_t)
				# batch_observation_k:[timesteps, state_dim=3]
				# batch_actions_k: [timesteps, action_dim=1]
				# curr_state_value: [timesteps]
				# curr_log_probs: [timesteps]
				curr_state_value, curr_log_probs = self.evaluate(batch_observation_k, batch_actions_k)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				# 重要性采样的权重, pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# importance_ratios: [timesteps]
				importance_ratios = torch.exp(curr_log_probs - batch_log_probs_k) # 使用log是为了计算数值稳定性

				# Calculate surrogate losses.
				# importance_ratios: [timesteps]
				# advantage_k: [timesteps]
				# surr1, surr2: [timesteps]
				surr1 = importance_ratios * advantage_k
				surr2 = torch.clamp(importance_ratios, min=1 - self.clip, max=1 + self.clip) * advantage_k # clip(x, 1-eps, 1+eps)
				surr_reward = torch.min(surr1, surr2)

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				# PPO - clip: 近端策略优化裁剪
				# actor_loss: 标量
				actor_loss = (-surr_reward).mean() # actor本来应该计算 最大化策略状态动作回报,因为optimizer的原因现在计算其最小值

				# 裁判critic的损失是与旧actor折扣回报进行比较,所以是mse loss
				# state_value: [timesteps]
				# batch_discount_reward: [timesteps]
				# critic_loss: 标量
				critic_loss = nn.MSELoss().forward(curr_state_value, batch_discount_reward_k) # batch_discount_reward_k是与环境交互得来，是相对真实的reward

				"""
				 在PyTorch中，当执行backward()时，若retain_graph=False，计算图会被释放，导致无法进行第二次反向传播。
				 retain_graph=True用于在有多个损失函数且需要分别计算梯度时，如神经网络风格迁移中Content Loss和Style Loss的情况，
				 防止中间变量(即激活值)被释放，确保两个loss的反向传播互不影响。
				"""
				# 计算梯度并对actor网络进行反向传播
				# Calculate gradients and perform backward propagation for actor network
				self.actor_optimizer.zero_grad()
				actor_loss.backward(retain_graph=True) # 注意：不能清空反向传播时的graph, 因为critic反向传播时还会使用, 原因是他们的梯度都会传到actor网络中去
				self.actor_optimizer.step()

				# 计算梯度并对critic网络进行反向传播
				# Calculate gradients and perform backward propagation for critic network
				self.critic_optimizer.zero_grad() # 最后一个 backward() 不要加 retain_graph 参数，这样每次更新完成后会释放占用的内存，也就不会出现越来越慢的情况了
				critic_loss.backward() # 注意：此时可以清空反向传播时的graph，与actor_loss一样，他们的梯度也会传到actor网络中去
				self.critic_optimizer.step()

				"""
				到最后actor loss, critic loss都下降了很多
				-------------------- Iteration #1294 --------------------
				Average Episodic Length: 200.0
				Average Episodic Return: -210.43
				Average actor Loss: -0.00221
				Average critic Loss: 83.57135
				Timesteps So Far: 2846800
				Iteration took: 1.92 secs
				"""
				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())
				self.logger['critic_losses'].append(critic_loss.detach())

			# Print a summary of our training so far
			self.show_summary()

			# Save our model if it's time
			if iter_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

	def get_actor_play_data(self):
		"""
			利用当前的actor策略模型与环境交互, 收集批量实验数据

			这就是我们从实验中收集⼀批数据的地⽅。由于这是⼀个on-policy的算法，我们需要在每次迭代 演员/裁判 ⽹络时收集⼀批新的数据。

			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_observation = [] # [timesteps, state_dim=3]
		batch_actions = [] # [timesteps, action_dim=1]
		batch_log_probs = [] # [timesteps]
		batch_rewards = [] # [episode_num, timesteps]
		batch_discount_reward = []# [timesteps]
		batch_lens = [] # [episode_num]

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		# ⼀局的数据。追踪每⼀局的奖励，在一局结束的时候会被清空，开始新的一局。
		episode_rewards = []

		# 追踪到⽬前为⽌这批程序我们已经运⾏了多少个时间段
		total_timestep = 0 # Keeps track of how many timesteps we've run so far this batch

		# 继续实验，直到我们每批运⾏超过或等于指定的时间步数
		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while total_timestep < self.timesteps_per_batch: # timestep_per_batch: 4800
			episode_rewards = [] # rewards collected per episode, 每回合收集的奖励

			# 重置环境
			# Reset the environment. Note that obs is short for observation.
			observation, info = self.env.reset() # observation:[state_dim=3]
			is_done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			timestep_in_episode=-1
			for timestep_in_episode in range(self.max_timesteps_per_episode): # 每一局棋最多下多少步, eg:200步
				# If render is specified, render the environment
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()
				# 递增时间步数
				total_timestep += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				# batch_observation:[timestep, state_dim=3]
				batch_observation.append(observation) # observation:[state_dim=3]

				# Calculate action and make a step in the env. observation:[state_dim=3], 3维向量
				# action:[action_dim=1]
				# log_prob:标量
				action, log_prob = self.get_action(observation) # 根据当前的actor模型的分布采样一个动作,以及生成该动作的log概率, action是一维向量
				# observation:[state_dim=1]
				# instant_reward:标量, 即时奖励来源于环境
				observation, instant_reward, is_done, is_truncated, _ = self.env.step(action)

				# Track recent reward, action, and action log probability
				# episode_rewards:[timestep]
				# batch_actions:[timestep, action_dim=3]
				# batch_log_prob:[timestep]
				episode_rewards.append(instant_reward) # 这个reward为即时奖励
				batch_actions.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if is_done: # 如一局游戏结束
					break

			# Track episodic lengths and rewards
			# batch_lens: [episode_num]
			# batch_rewards: [episode_num, timesteps]
			batch_lens.append(timestep_in_episode + 1)
			batch_rewards.append(episode_rewards)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_observation = torch.tensor(batch_observation, dtype=torch.float)
		batch_actions = torch.tensor(batch_actions, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

		#计算当前以及未来的折扣的回报
		# batch_rewards: [episode_num, timesteps]
		# batch_reward_to_go: [episode_num*timesteps]
		batch_discount_reward = self.compute_discount_rewards(batch_rewards)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rewards
		self.logger['batch_lens'] = batch_lens

		return batch_observation, batch_actions, batch_log_probs, batch_discount_reward, batch_lens

	def compute_discount_rewards(self, batch_rewards:List[List[float]])->torch.Tensor:
		"""
			计算当前以及未来的所有的回报, 即折扣回报
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rewards - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for episode_rewards in reversed(batch_rewards): # [number of episodes, number of timesteps per episode]
			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for current_reward in reversed(episode_rewards):
				discounted_reward = current_reward + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		# 将每个回合的折扣奖励的数据转换成张量
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
		return batch_rtgs

	def get_action(self, observation:np.array):
		"""
			根据当前的状态以及actor模型，预测当前需要采取的动作

			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# 梯度回传：log_prob -> dist -> mean -> actor
		# Query the actor network for a mean action
		mean = self.actor.forward(observation)

		# 用上述协方差矩阵中的平均行动和标准差创建一个分布。
		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		# 从多维正态分布中采样
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		sample_action = dist.sample() # sample是no_grad()上下文中的

		# Calculate the log probability for that action
		log_prob = dist.log_prob(sample_action) # Returns the log of the probability density/mass function evaluated at value.

		# Return the sampled action and the log probability of that action in our distribution
		return sample_action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_observations:torch.Tensor, batch_actions:torch.Tensor):
		"""
            评个每个状态的Q值，以及最近一批actor网络迭代中的每个action的对数prob。

			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""

		# 使用当前Critic网络评价当前价值网络的状态值
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		# batch_observations:[timestep, state_dim=3]
		# state_value:[timestep, 1] -> [timestep]
		state_value = self.critic.forward(batch_observations).squeeze()

		# 使用当前actor网络生成均值，作为多维正态分布的均值， 从分布中计算batch_action的真实对数概率
		# 下面是采用了重参数技巧(reparameterization trick), 梯度从 log_action_probs -> distribute -> mean -> actor
		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor.forward(batch_observations)
		distribute = MultivariateNormal(mean, self.cov_mat) # 多维正态分布, 由于是连续值，因此需要从多维正态分布中采样
		# batch_action:[timestep, action_dim=1]
		log_action_probs = distribute.log_prob(batch_actions) # log_prob returns the logarithm of the density or probability, 计算概率的log

		# 返回批次中每个观察值的值向量V和批次中每个动作的对数概率log_probs
		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return state_value, log_action_probs

	def _init_hyperparameters(self, hyperparameters:Dict[str, Any]):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout(卷展栏；首次展示)
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 100                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def show_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_time = self.logger['delta_time']
		self.logger['delta_time'] = time.time_ns()
		delta_time = (self.logger['delta_time'] - delta_time) / 1e9
		delta_time = str(round(delta_time, 2))

		timestep_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_episode_lens = np.mean(self.logger['batch_lens'])
		avg_episode_rewards = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_episode_lens = str(round(avg_episode_lens, 2))
		avg_episode_rewards = str(round(avg_episode_rewards, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))
		avg_critic_loss = str(round(avg_critic_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_episode_lens}", flush=True)
		print(f"Average Episodic Return: {avg_episode_rewards}", flush=True)
		print(f"Average actor Loss: {avg_actor_loss}", flush=True)
		print(f"Average critic Loss: {avg_critic_loss}", flush=True)
		print(f"Timesteps So Far: {timestep_so_far}", flush=True)
		print(f"Iteration took: {delta_time} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['critic_losses'] = []
