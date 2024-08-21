#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # TO DO: Add own code
            best_action = argmax(self.Q_sa[s])
            probabilities = np.ones(self.n_actions) * epsilon/self.n_actions
            probabilities[best_action] = 1 - epsilon * (self.n_actions-1)/self.n_actions
            # print(probabilities)

            a = np.random.choice(self.n_actions, p = probabilities) # Replace this with correct action selection
            # print(a)
            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # TO DO: Add own code
            
            probabilities = softmax(self.Q_sa[s],temp)
            # print(probabilities)
            a = np.random.choice(self.n_actions, p = probabilities) # Replace this with correct action selection
            # print(a)
            
        return a
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        
        T = len(states)-1
        G = 0
        for i in range(T-1,0,-1):
            G = rewards[i] + self.gamma * G
            self.Q_sa[states[i],actions[i]] = self.Q_sa[states[i],actions[i]] + self.learning_rate * (G - self.Q_sa[states[i],actions[i]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, validate=False,interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    validate_mean_rewards_per_timestep = []

    count = 0
    while(count < n_timesteps):
        s = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards =   []
        episode_states.append(s)
        for t in range(max_episode_length):
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp = temp)
            s,r,done = env.step(a)
            count +=1
            if(validate and count % interval == interval-1):
                mean_reward_per_timestep = greedy_simulate(pi,150)
                validate_mean_rewards_per_timestep.append(mean_reward_per_timestep)

            rewards.append(r)
            episode_actions.append(a)
            episode_states.append(s)
            episode_rewards.append(r)
            if(count >= n_timesteps):
                break
            if done:
                break
            

        
        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=1) # Plot the Q-value estimates during Q-learning execution

        
        pi.update(episode_states,episode_actions,episode_rewards)

    if(validate):
        return validate_mean_rewards_per_timestep
        
    return rewards 

def greedy_simulate(pi,max_timestemp = 150):
    validate_env = StochasticWindyGridworld(initialize_model=False)
    rewards = []
    s = validate_env.reset()
    for _ in range(max_timestemp):
        a = argmax(pi.Q_sa[s])
        s,r,done = validate_env.step(a)
        rewards.append(r)
        if(done):
            break
    return np.array(rewards).mean()  
    
def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.001

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False
    for _ in range(100):
        rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                    policy, epsilon, temp, plot,validate=True)
        print("Obtained rewards: {}".format(len(rewards))) 

            
if __name__ == '__main__':
    test()


