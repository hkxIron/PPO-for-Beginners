""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
"""
https://karpathy.github.io/2016/05/31/rl/

andrew karpathy
"""

import numpy as np
import pickle
import gym

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization, w/sqrt(f_in)
    model['W2'] = np.random.randn(H) / np.sqrt(H) # w/sqrt(f_in)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def prepro(I:np.array):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    #print(I)
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float32).ravel()


def discount_rewards(r:np.array)->np.array:
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = r[t] + running_add * gamma # 当前的收益加上未来拆扣的收益
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x:np.array):
    # p=sigmoid(W2*relu(W1*x))
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp) #  # compute log probability of going up,计算球拍向上移动的概率
    return p, h  # return probability of taking action 2, and hidden state


# 计算参数梯度
def policy_backward(episode_hidden_states, d_episode_logprobs):
    """ backward pass. (eph is array of intermediate hidden states) """
    # p=sigmoid(W2*relu(W1*x))
    dW2 = np.dot(episode_hidden_states.T, d_episode_logprobs).ravel()
    dh = np.outer(d_episode_logprobs, model['W2'])
    dh[episode_hidden_states <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, episode_xs)
    return {'W1': dW1, 'W2': dW2}


print("all_envs:", gym.envs.registry.keys())
# 请先安装包依赖： pip install "gym[atari, accept-rom-license]"
env = gym.make("ALE/Pong-v5") # 打乒乓球
#env = gym.make("Pong-v1") # 打乒乓球
#env = gym.make("Pendulum-v1") # 打乒乓球
#env = gym.make("LunarLander-v2") # 打乒乓球

observation, info = env.reset()
prev_x = None  # used in computing the difference frame
xs, hidden_states, dlogps, episode_reward_list = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    action_prob, hidden_state = policy_forward(x) # 预测采取不同动作的概率

    # action只能取两个值：2和3
    action = 2 if np.random.uniform() < action_prob else 3  # roll the dice, 骰子

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hidden_states.append(hidden_state)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(y - action_prob) # mse差值

    # step the environment and get new measurements
    observation, reward, episode_is_done, truncated, info = env.step(action)
    reward_sum += reward

    episode_reward_list.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if episode_is_done:  # an episode finished,  一局游戏是否结束
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        episode_xs = np.vstack(xs)
        episode_hidden_states = np.vstack(hidden_states)
        episode_dlogp = np.vstack(dlogps)
        episode_rewards = np.vstack(episode_reward_list)
        xs, hidden_states, dlogps, episode_reward_list = [], [], [], []  # reset array memory

        # 一局游戏结束了，计算折扣reward
        # compute the discounted reward backwards through time
        discounted_eposide_reward = discount_rewards(episode_rewards)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_eposide_reward -= np.mean(discounted_eposide_reward) # 减去该局游戏中的均值
        discounted_eposide_reward /= np.std(discounted_eposide_reward) # 除以该局游戏中的方差

        # policy gradient:
        # grad(x) = discounted_reward * dlogp(x)
        episode_dlogp *= discounted_eposide_reward  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(episode_hidden_states, episode_dlogp)

        for k,v in model.items():
            grad_buffer[k] += grad[k]  # accumulate grad over batch for rmsprop

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))

        reward_sum = 0
        observation, info = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: game finished, reward: %f' % (episode_number, reward))