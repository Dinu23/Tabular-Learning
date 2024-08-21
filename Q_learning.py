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
from Helper import softmax, argmax, linear_anneal

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.N_sa = np.ones((n_states,n_actions))

    def select_action(self, s, t, policy='egreedy', epsilon=None, temp=None, C=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            
            # TO DO: Add own code
            best_action = argmax(self.Q_sa[s])
            probabilities = np.ones(self.n_actions) * epsilon/self.n_actions
            probabilities[best_action] = 1 - epsilon * (self.n_actions-1)/self.n_actions
        
            a = np.random.choice(self.n_actions, p = probabilities) 

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # TO DO: Add own code
            probabilities = softmax(self.Q_sa[s],temp)
           
            a = np.random.choice(self.n_actions, p = probabilities) 

        elif policy == 'UCB1':
            if C is None:
                raise KeyError("Provide C")
            a = argmax(self.Q_sa[s] + C * np.sqrt(np.log(t+1)/self.N_sa[s]))
           
        self.N_sa[s,a] +=1    
        return a
        
    def update(self,s,a,r,s_next,done):
        G = r + self.gamma * np.max(self.Q_sa[s_next])
        self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (G - self.Q_sa[s,a])

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, C=None, plot=True, policy_annealing=False, policy_linear_annealing = None, lr_annealing = False, lr_linear_annealing = None, validate = False, interval = 500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    validate_mean_rewards_per_timestep = []

    # TO DO: Write your Q-learning algorithm here!
    s = env.reset()
    for t in range(n_timesteps):
        if(policy_annealing):
            if(policy_linear_annealing == None):
                raise KeyError("Provide policy linear annealing")
            epsilon = policy_linear_annealing.get_value(t,n_timesteps)
            temp = policy_linear_annealing.get_value(t,n_timesteps)
            C = policy_linear_annealing.get_value(t,n_timesteps)

        if(lr_annealing):
            if(lr_linear_annealing == None):
                raise KeyError("Provide learning rate linear annealing")
            lr = lr_linear_annealing.get_value(t,n_timesteps)
            pi.learning_rate = lr 

        a = pi.select_action(s,t,policy=policy,epsilon=epsilon,temp = temp, C=C)
        s_next,r,done = env.step(a)
        if plot:
           env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

        rewards.append(r)
        pi.update(s,a,r,s_next,done)

        if(done):
            s = env.reset()
        else:   
            s = s_next

        if(validate and t % interval == interval-1):
            mean_reward_per_timestep = greedy_simulate(pi,150)
    
            validate_mean_rewards_per_timestep.append(mean_reward_per_timestep)
    
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
    
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1
    
    # Exploration
    policy = 'UCB1' # 'egreedy' or 'softmax' 
    epsilon = 0.4
    temp = 1.0
    C=1
    policy_annealing = True
    policy_linear_annealing = Linear_anneal(0.3,0.02,80/100)
    lr_annealing = True
    lr_linear_annealing = Linear_anneal(0.3,0.02,80/100) 
    # Plotting parameters
    plot = False

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, C, plot,validate=True)
    print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()


class Linear_anneal:
    def __init__(self, start, final, percentage):
        self.start = start
        self.final = final
        self.percentage = percentage
    
    def get_value(self,t,T):
        return linear_anneal(t, T, self.start, self.final, self.percentage)

