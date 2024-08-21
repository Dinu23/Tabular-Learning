#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        
        a = argmax(self.Q_sa[s]) 
        
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''

        self.Q_sa[s, a] = np.sum(p_sas  * (r_sas + self.gamma * np.max(self.Q_sa,axis = 1))) 

    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
        
    max_error = np.inf
    i = 0
    while(max_error > threshold):
        max_error = 0
        i += 1
        for s in range(env.n_states):
            for a in range(env.n_actions):
                current_value = QIagent.Q_sa[s, a]
                p_sas, r_sas = env.model(s, a)
                QIagent.update(s, a, p_sas, r_sas)
                max_error = max(max_error, abs(QIagent.Q_sa[s, a] - current_value))

        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.00001)
        print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
         
    
     
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # View optimal policy
    done = False
    s = env.reset()
    count = 0
    overall_reward = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        count +=1
        overall_reward += r
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.00001)
        s = s_next
    mean_reward_per_timestep_per_iteration= overall_reward / count
    # TO DO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep_per_iteration))

    mean_reward_per_timestep = [mean_reward_per_timestep_per_iteration]
    repetitions = 50
    for _ in range(repetitions-1):
        done = False
        s = env.reset()
        count = 0
        overall_reward = 0
        while not done:
            a = QIagent.select_action(s)
            s_next, r, done = env.step(a)
            count +=1
            overall_reward += r
            # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=1)
            s = s_next
        mean_reward_per_timestep_per_iteration = overall_reward/ count
        mean_reward_per_timestep.append(mean_reward_per_timestep_per_iteration)
    
    mean_reward_per_timestep = np.array(mean_reward_per_timestep)
    mean_reward = mean_reward_per_timestep.mean()
    std_reward = mean_reward_per_timestep.std()
    print("Mean and STD reward per timestep under optimal policy: {}, {} over {} repetitions".format(mean_reward,std_reward, repetitions))
    print("V*(s = 3) = {}".format(np.max(QIagent.Q_sa[3])) )


if __name__ == '__main__':
    experiment()

