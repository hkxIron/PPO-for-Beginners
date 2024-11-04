
from typing import *

import gym
import sys
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import math
from functools import partial


"""
包依赖安装:
sudo apt install swig
pip3 install box2d box2d-kengz

用如下代码训练后，
到300个episode之后，确实average reward减小了很多, 
460个episode之后，reward开始为正

Episode 10	Last reward: -86.61	Average reward: -131.26
...
Episode 300	Last reward: -95.37	Average reward: -24.50
...
Episode 460	Last reward: -25.21	Average reward: 10.39
...
Episode 480	Last reward: 50.65	Average reward: 35.91




下面是一个简单的策略网络，
案例：模拟登月小艇降落在月球表面时的情形。任务的目标是让登月小艇安全地降落在两个黄色旗帜间的平地上。测试环境：LunarLander-v2

Obs：这个游戏环境有八个观测值，分别是水平坐标x，垂直坐标y，水平速度，垂直速度，角度，角速度，腿1触地，腿2触地；

Action：agent可以采取四种离散行动，分别是什么都不做，发动左方向引擎喷射，发动主引擎向下喷射，发动右方向引擎喷射。

Reward：小艇坠毁得-100分；小艇成功着陆在两个黄色旗帜之间得100~140分；喷射主引擎向下喷火每次得-0.3分；小艇最终完全静止则再得100分；每条腿着地各得10分。

这里我们虽然采用的是离散的动作空间，但是整体代码是相差不大的，感兴趣的同学可以尝试下连续的动作空间。

blog详见：
https://www.cnblogs.com/xingzheai/p/15826847.html
"""

class PolicyNet(nn.Module):
    def __init__(self, n_states_num:int, n_actions_num:int, hidden_size:int):
        super(PolicyNet, self).__init__()
        self.data = []  # 存储(reward, p(action|state))轨迹
        # 输入为长度为8的向量 输出为4个动作
        self.net = nn.Sequential(
            # 两个线性层，中间使用Relu激活函数连接，最后连接softmax输出每个动作的概率
            nn.Linear(in_features=n_states_num, out_features=hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=n_actions_num, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x:torch.Tensor):
        # 状态输入s的shape为向量：[batch, n_state_num]
        action_prob = self.net(x) # shape: [batch, n_actions_num]
        return action_prob

def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int=1
):
    # 1. 如果current_step< num_warmup_steps,则从0一直增长，直到1,此时lr=lambda*init_lr=1*init_lr=init_lr
    # 2. 如果current_step>num_warmup_steps,则使用cos学习率进行减小直到0，当然可以控制cos循环次数
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

class PolicyGradient():
    def __init__(self, n_states_num:int=8, n_actions_num:int=4, base_learning_rate=0.01, reward_decay=0.95):
        # 状态数   state是一个8维向量，分别是水平坐标x,垂直坐标y,水平速度,垂直速度,角度,角速度,腿1触地,腿2触地
        self.n_states_num = n_states_num
        # action是4维、离散，即什么都不做，发动左方向引擎，发动主机，发动右方向引擎。
        self.n_actions_num = n_actions_num
        # 学习率
        self.lr = base_learning_rate
        # gamma
        self.gamma = reward_decay
        # 策略网络
        self.action_policy = PolicyNet(n_states_num, n_actions_num, 64)
        # 优化器
        self.optimizer = torch.optim.Adam(self.action_policy.parameters(), lr=base_learning_rate, betas=(0.9, 0.98), eps=1e-9)
        lambda_func = partial(_get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda, num_warmup_steps=200, num_training_steps=2*10000, num_cycles=0.5)
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                     lr_lambda=lambda_func)
        # 存储轨迹  存储数据为（每一次的reward，动作的概率）
        self.data = []
        self.loss_history = []

    # 存储轨迹数据
    def record_reward_and_action_prob(self, item:Tuple[float, torch.Tensor]):
        # 记录r,log_P(a|s)
        self.data.append(item)

    def train_net_by_episode(self):
        # 计算梯度并更新策略网络参数。tape为梯度记录器
        discount_reward = 0  # 终结状态的初始回报为0
        policy_loss = []
        for instant_reward, log_prob in self.data[::-1]:  # 逆序遍历
            discount_reward = instant_reward + self.gamma * discount_reward  # 计算每个时间戳上的回报
            # 当前步的reward为产生动作的概率与回报之积
            step_reward = log_prob * discount_reward
            # 每个时间戳都计算一次梯度
            loss = -step_reward
            policy_loss.append(loss)
        # ----------------
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()  # 求和
        # print('policy_loss:', policy_loss.item())
        # 反向传播
        policy_loss.backward()
        self.optimizer.step()
        self.loss_history.append(policy_loss.item())
        # print('cost_his:', self.cost_his)
        self.data = []  # 清空轨迹

    # 将状态传入神经网络 根据概率选择动作
    def choose_action(self, state:np.array)->Tuple[int, torch.Tensor]:
        # 将state转化成tensor 并且维度转化为[8]->[1,8]
        s = torch.Tensor(state).unsqueeze(0)
        action_prob = self.action_policy(s)  # 动作分布:[0,1,2,3]
        # 从类别分布中采样1个动作, shape: [1] torch.log(prob), 1

        # 作用是创建以参数prob为标准的类别分布，样本是来自“0 … K-1”的整数，其中K是prob参数的长度。也就是说，按照传入的prob中给定的概率，
        # 在相应的位置处进行取样，取样返回的是该位置的整数索引。不是最大的，是按照概率采样的那个，采样到那个就是哪个的索引
        action_sampler = torch.distributions.Categorical(action_prob)  # 生成分布
        action = action_sampler.sample()
        return action.item(), action_sampler.log_prob(action)

    def plot_cost(self, average_reward):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(average_reward)), average_reward)
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        plt.show()

def train():
    env = gym.make('LunarLander-v2')
    print_interval = 100
    policy_gradient = PolicyGradient(n_states_num=8, n_actions_num=4, base_learning_rate=0.01)
    average_reward:List[float] = []
    running_reward=None

    for n_episode in range(2*10000): # 玩1000局游戏
        state, _ = env.reset()  # 回到游戏初始状态，返回s0
        episode_reward = 0
        for t in range(10001):  # CartPole-v1 forced to terminates at 1000 step.每局游戏下1001步
            # 根据状态 传入神经网络 选择动作
            action, log_prob = policy_gradient.choose_action(state)
            # print(action, log_prob)
            # 与环境交互
            new_state, reward, done, truncated, info = env.step(action)
            # s_prime, reward, done, info = env.step(action)
            if n_episode > 10000:
                env.render()
            # 记录动作a和动作产生的奖励r
            policy_gradient.record_reward_and_action_prob((reward, log_prob))
            state = new_state  # 刷新状态
            episode_reward += reward
            if done:  # 当前episode终止
                break
            # episode终止后，训练一次网络
        # ------------------------------
        if running_reward is not None:
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        else:
            running_reward = episode_reward
        average_reward.append(running_reward)

        # 交互完成后 进行学习
        policy_gradient.train_net_by_episode()
        policy_gradient.lr_scheduler.step() # 重新计算lr步长

        if n_episode % print_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f} lr:{:.6f}'.format( n_episode, episode_reward, running_reward, policy_gradient.lr_scheduler.get_last_lr()[0]))
        if running_reward > env.spec.reward_threshold:  # 大于游戏的最大阈值时，退出游戏
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            torch.save(policy_gradient.action_policy.state_dict(), 'pg.pkl')
            break

    # 所有episode训练完，画一下reward图
    policy_gradient.plot_cost(average_reward)

if __name__ == '__main__':
    train()