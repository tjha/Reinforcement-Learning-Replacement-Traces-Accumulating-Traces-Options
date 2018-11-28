# Tejas Jha
# EECS 498 Reinforcement Learning HW 4
######################################################################################
# This python script contains all of the relevant code needed to recreate the desired
# plots on the assignment. Comments will spearate the code to better indcated which 
# protions corresponding to which parts of the assignment.

import gym
from gym.envs.registration import register
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import floor, log
from itertools import zip_longest
import tiles3 as t3
from diagram import MAPS
from diagram import plot_gridworld

# Environment Generation
register(
    id = 'MountainCar-v1',
    entry_point = 'gym.envs.classic_control:MountainCarEnv',
    max_episode_steps = 5000
)

env = gym.make('MountainCar-v1')
###########################################################################################
#
# cp code
# all possible actions
ACTION_REVERSE = 0
ACTION_ZERO = 1
ACTION_FORWARD = 2
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# discount is always 1.0 in these experiments
DISCOUNT = 1.0

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# maximum steps per episode
STEP_LIMIT = 5000

# take an @action at @position and @velocity
# @return: new position, new velocity, reward (always -1)
def step(position, velocity, action):
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(VELOCITY_MIN, new_velocity), VELOCITY_MAX)
    new_position = position + new_velocity
    new_position = min(max(POSITION_MIN, new_position), POSITION_MAX)
    reward = -1.0
    if new_position == POSITION_MIN:
        new_velocity = 0.0
    return new_position, new_velocity, reward


def accumulating_trace(trace, active_tiles, lam):
    trace *= lam * DISCOUNT
    trace[active_tiles] += 1
    return trace

def replacing_trace(trace, activeTiles, lam):
    active = np.in1d(np.arange(len(trace)), activeTiles)
    trace[active] = 1
    trace[~active] *= lam * DISCOUNT
    return trace

class Sarsa:

    def __init__(self, step_size, lam, trace_update=accumulating_trace, num_of_tilings=8, max_size=4096):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.trace_update = trace_update
        self.lam = lam


        self.step_size = step_size / num_of_tilings
        self.hash_table = t3.IHT(max_size)
        self.weights = np.zeros(max_size)
        self.trace = np.zeros(max_size)

        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    def get_active_tiles(self, position, velocity, action):
        active_tiles = t3.tiles(self.hash_table, self.num_of_tilings,
                            [self.position_scale * position, self.velocity_scale * velocity],
                            [action])
        return active_tiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = target - estimation
        if self.trace_update == accumulating_trace or self.trace_update == replacing_trace:
            self.trace_update(self.trace, active_tiles, self.lam)
        else:
            raise Exception('Unexpected Trace Type')
        self.weights += self.step_size * delta * self.trace

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(position, velocity, valueFunction):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(position, velocity, action))
    return np.argmax(values)

# play Mountain Car for one episode based on given method @evaluator
# @return: total steps in this episode
def play(evaluator):
    position = np.random.uniform(-0.6, -0.4)
    velocity = 0.0
    action = get_action(position, velocity, evaluator)
    steps = 0
    while True:
        next_position, next_velocity, reward = step(position, velocity, action)
        next_action = get_action(next_position, next_velocity, evaluator)
        steps += 1
        target = reward + DISCOUNT * evaluator.value(next_position, next_velocity, next_action)
        evaluator.learn(position, velocity, action, target)
        position = next_position
        velocity = next_velocity
        action = next_action
        if next_position == POSITION_MAX:
            break
        if steps >= STEP_LIMIT:
            print('Step Limit Exceeded!')
            break
    return steps

def execute(evaluator):
    observations = env.reset()
    position = observations[0]
    velocity = observations[1]
    action = get_action(position, velocity, evaluator)
    steps = 0
    while True:
        env.render()
        observation, reward, done, _ = env.step(action)
        next_position = observation[0]
        next_velocity = observation[1]
        next_action = get_action(next_position, next_velocity, evaluator)
        steps += 1
        target = reward + DISCOUNT * evaluator.value(next_position, next_velocity, next_action)
        evaluator.learn(position, velocity, action, target)
        position = next_position
        velocity = next_velocity
        action = next_action
        if done:
            break
        if steps >= STEP_LIMIT:
            print('Step Limit Exceeded!')
            break
    return steps

# end cp code
###########################################################################################

# Question 1 plots
def q1plots():
    runs = 5
    episodes = 50
    lambdas = [0,0.9]

    # Part (b) - Generation of plot using replacing traces
    alphas = np.arange(0.6,2.0,0.2) / 8.0
    steps = np.zeros((len(lambdas), len(alphas), runs, episodes))
    for lambdaIdx, lam in enumerate(lambdas):
        for alphaIdx, alpha in enumerate(alphas):
            for run in tqdm(range(runs)):
                env.reset()
                evaluator = Sarsa(alpha, lam, replacing_trace, max_size=4096)
                for ep in range(episodes):
                    #step = play(evaluator)
                    step = execute(evaluator)
                    steps[lambdaIdx, alphaIdx, run, ep] = step

    # average over episodes
    steps = np.mean(steps, axis=3)
    # average over runs
    steps = np.mean(steps, axis=2)

    for lamdaIdx, lam in enumerate(lambdas):
        plt.plot(alphas, steps[lamdaIdx, :], label='lambda = %s' % (str(lam)))
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged steps per episode')
    plt.title('Sarsa with replacing traces')
    #plt.ylim([180, 300])
    plt.legend()

    plt.savefig('replacing_traces2.png')
    plt.close()

    print("Completed Problem 1 Part (b)")

    # Part (c) -    Generation of plot using accumulating traces
    alphas = np.arange(0.2,0.5,0.05) / 8.0
    steps = np.zeros((len(lambdas), len(alphas), runs, episodes))
    for lambdaIdx, lam in enumerate(lambdas):
        for alphaIdx, alpha in enumerate(alphas):
            for run in tqdm(range(runs)):
                evaluator = Sarsa(alpha, lam, accumulating_trace, max_size=4096)
                for ep in range(episodes):
                    #step = play(evaluator)
                    step = execute(evaluator)
                    steps[lambdaIdx, alphaIdx, run, ep] = step

    # average over episodes
    steps = np.mean(steps, axis=3)
    # average over runs
    steps = np.mean(steps, axis=2)
    
    for lamdaIdx, lam in enumerate(lambdas):
        plt.plot(alphas, steps[lamdaIdx, :], label='lambda = %s' % (str(lam)))
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged steps per episode')
    plt.title('Sarsa with accumulating traces')
    #plt.ylim([180, 300])
    plt.legend()

    plt.savefig('accumulating_traces2.png')
    plt.close()

    print("Completed Problem 1 Part (c)")

####################################################################################
# 
# Work for Q2

class ShortCorridor:
    """
    Short corridor environment, see Example 13.1
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0

    def step(self, go_right):
        """
        Args:
            go_right (bool): chosen action
        Returns:
            tuple of (reward, episode terminated?)
        """
        if self.state == 0 or self.state == 2:
            if go_right:
                self.state += 1
            else:
                self.state = max(0, self.state - 1)
        else:
            if go_right:
                self.state -= 1
            else:
                self.state += 1

        if self.state == 3:
            # terminal state
            return 0, True
        else:
            return -1, False

def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)

class ReinforceAgent:
    """
    ReinforceAgent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma):
        # set values such that initial conditions correspond to left-epsilon greedy
        self.theta = np.array([-1.47, 1.47])
        self.alpha = alpha
        self.gamma = gamma
        # first column - left, second - right
        self.x = np.array([[0, 1],
                           [1, 0]])
        self.rewards = []
        self.actions = []

    def get_pi(self):
        h = np.dot(self.theta, self.x)
        t = np.exp(h - np.max(h))
        pmf = t / np.sum(t)
        # never become deterministic,
        # guarantees episode finish
        imin = np.argmin(pmf)
        epsilon = 0.05

        if pmf[imin] < epsilon:
            pmf[:] = 1 - epsilon
            pmf[imin] = epsilon

        return pmf

    def get_p_right(self):
        return self.get_pi()[1]

    def choose_action(self, reward):
        if reward is not None:
            self.rewards.append(reward)

        pmf = self.get_pi()
        go_right = np.random.uniform() <= pmf[1]
        self.actions.append(go_right)

        return go_right

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * G[i] * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        super(ReinforceBaselineAgent, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w
        self.w = 0

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            self.w += self.alpha_w * gamma_pow * (G[i] - self.w)

            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * (G[i] - self.w) * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

def trial(num_episodes, agent_generator):
    env = ShortCorridor()
    agent = agent_generator()

    rewards = np.zeros(num_episodes)
    for episode_idx in range(num_episodes):
        rewards_sum = 0
        reward = None
        env.reset()

        while True:
            go_right = agent.choose_action(reward)
            reward, episode_end = env.step(go_right)
            rewards_sum += reward

            if episode_end:
                agent.episode_end(reward)
                break

        rewards[episode_idx] = rewards_sum

    return rewards

def q2plot1():
    num_trials = 100
    num_episodes = 1000
    alphas = [2**(-12), 2**(-13), 2**(-14)]
    gamma = 1

    for alpha in alphas:

        print(alpha)
        
        rewards = np.zeros((num_trials, num_episodes))
        agent_generator = lambda : ReinforceAgent(alpha=alpha, gamma=gamma)

        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[i, :] = reward

        plt.plot(np.arange(num_episodes) + 1, rewards.mean(axis=0), label='alpha = %s' % (str(alpha)))

    #plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    #plt.plot(np.arange(num_episodes) + 1, rewards.mean(axis=0), color='blue')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('fig1_2.png')
    plt.close()

def q2plot2():
    num_trials = 100
    num_episodes = 1000
    alpha = 2**(-13)
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceBaselineAgent(alpha=2**(-9), gamma=gamma, alpha_w=2**(-6))]
    labels = ['Reinforce with baseline',
              'Reinforce without baseline']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    #plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('fig2_2.png')
    plt.close()

#################################################################################################################
#
# Code for Q3

class FourRooms:
    """
    Four Rooms environment from Options paper
    Weird implementation for planning purposes
    """
    def __init__(self, start_h, start_w):
        self.reset(start_h, start_w)

    def reset(self, start_h, start_w):
        self.goal_state = (start_h, start_w)

    def action_taken(self, action):
        if np.random.binomial(1, 2/3) == 1:
            return action
        possible_actions = [0,1,2,3]
        possible_actions.pop(action)
        return np.random.choice(possible_actions)

    def step_reward(self, current_state, direction):
        """
        Args:
            current_state: tuple corresponding to location
            direction: up(0), right(1), down(2), left(3)
        Returns:
            tuple of (reward, episode terminated?)
        """
        new_state = current_state.copy()
        action = self.action_taken(direction)

        if action == 0 and new_state[0] != 1:
            new_state[0] -= 1
        elif action == 1 and new_state[1] != 11:
            new_state[1] += 1
        elif action == 2 and new_state[0] != 11:
            new_state[0] += 1
        elif action == 3 and new_state[1] != 1:
            new_state[1] -= 1

        if new_state == self.goal_state:
            # terminal state
            return 1, True
        else:
            return 0, False

HALLWAYS = [(6,2), (3,6), (7,9), (10,6)]

class Option:
    def __init__(self, policy, termination_condition, initiation_set):
        self.policy = policy
        self.goal_state = termination_condition
        self.possible_states = initiation_set
    
    def valid(self, state):
        if state in self.possible_states:
            return True
        else:
            return False

    def get_action(self, state):
        if self.valid(state):
            return self.policy[state]
        else:
            return -1
    
    def target(self):
        return self.goal_state

class Planner:
    "performs planning iterations"
    def __init__(self, goal_state, options):
        self.V = {goal_state:1}
        self.options = options

def q3fig4():
    print("Printing primitive Options")
    # Print a diagram for the resulting V from initial to the second iteration
    V = {(7,9):1}
    plot_gridworld(MAPS,V, 1)

    env4 = FourRooms(7,9)


def q3fig5():
    print("heyoo")


if __name__ == '__main__':

    #plot_gridworld(MAPS, V)
    # Ensuring environment is reset to begin 
    #env.reset()
    # -------------------------------------------------------------------------------
    # Question 1 -  Implementation of replacing traces and accumulating traces
    #               to produce plots similar to that of Figure 12.10 in the textbook
    # -------------------------------------------------------------------------------
    q1plots()

    #---------------------------------------------------------------------------------
    # Quesiton 2 -  Reproduction of Figures 13.1 and 13.2 from the textbook.
    #               Plots are generated using example 13.1
    # --------------------------------------------------------------------------------
    # Figure 13.1 Reproduciton
    #q2plot1()
    # Figure 13.2 Reproduction
    #q2plot2()

    #---------------------------------------------------------------------------------
    # Quesiton 3 -  Attempts to reproduce the work behind Figures 4 and 5 in the 
    #               Options paper. Heatmap figure will be created to represent the results
    # --------------------------------------------------------------------------------
    # Generation of Figure 4
    q3fig4()
    # Generation of Figure 5
    q3fig5()


    # Ensuring environment is closed at the end to avoid compilation issues
    env.close()