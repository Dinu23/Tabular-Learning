#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time

from Q_learning import q_learning
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import LearningCurvePlot, smooth, linear_anneal

class MyLearningCurvePlot(LearningCurvePlot):
    def add_curve_validate(self,x,y,label=None):
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, C=None, smoothing_window=51, plot=False, n=5,  policy_annealing=False, policy_linear_annealing = None,
                    lr_annealing = False, lr_linear_annealing = None,validate = False,interval=500):
    if(validate):
        reward_results = np.empty([n_repetitions,np.int32(n_timesteps/interval)]) # Result array
    else:
        reward_results = np.empty([n_repetitions,n_timesteps]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, C,
                                 plot, policy_annealing, policy_linear_annealing, lr_annealing,
                                   lr_linear_annealing, validate = validate,interval =interval)
        elif backup == 'sarsa':
            rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp,
                             plot, policy_annealing, policy_linear_annealing, lr_annealing,
                               lr_linear_annealing, validate = validate,interval =interval)
        elif backup == 'mc':
            rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, validate = validate,interval =interval)
        elif backup == 'nstep':
            rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n, validate = validate,interval =interval)

        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))  
      
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve
    

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 50
    smoothing_window = 1001
    smoothing_window_validate = 5
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0
        
    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    
    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_average_reward_per_timestep = 1.3033392415977556 # set the optimal average reward per timestep you found in the DP assignment here
    
    #### Assignment 2: Effect of exploration
    a2_exploration(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep)
    ### Assignment 2: Effect of exploration with validation every 500 steps
    a2_exploration_validate(n_repetitions, smoothing_window_validate, backup_labels, plot, n_timesteps, max_episode_length, gamma,optimal_average_reward_per_timestep)
    

    # ###### Assignment 3: Q-learning versus SARSA
    a3_on_off_policy(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep)
    # ###### Assignment 3: Q-learning versus SARSA with validation every 500 steps
    a3_on_off_policy_validate(n_repetitions, smoothing_window_validate, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep)
    
    # ##### Assignment 4: Back-up depth
    a4_depth(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep)
    # ##### Assignment 4: Back-up depth with validation every 500 steps
    a4_depth_validate(n_repetitions, smoothing_window_validate, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep)

def a2_exploration(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep):
    n = 5
    temp = 1.0
    epsilon = 0.3
    C = 0.1

    policy = 'egreedy'
    epsilons = [0.02,0.1,0.3]
    learning_rate = 0.25
    backup = 'q'
    Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration versus UCB exploration')    
    for epsilon in epsilons:       
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
    
    policy_annealing = True
    policy_linear_annealing = Linear_anneal(0.3,0.02,80/100)
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n, policy_annealing, policy_linear_annealing)
    Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $- annealing')

    policy = 'softmax'
    temps = [0.01,0.1,1.0]
    for temp in temps:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))

    policy_annealing = True
    policy_linear_annealing = Linear_anneal(1,0.01,80/100)
    
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n, policy_annealing, policy_linear_annealing)
    Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ - annealing')


    policy = 'UCB1'
    Cs = [1,5]
    for C in Cs:       
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'UCB1, C = {} '.format(C))

    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('exploration_training_reward.png')

def a2_exploration_validate(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep):
    n = 5
    temp = 1.0
    epsilon = 0.3
    C = 0.1

    interval = 500
    validation_points = np.array([interval*(i+1) for i in range(np.int32(n_timesteps/interval))])
    
    policy = 'egreedy'
    epsilons = [0.02,0.1,0.3]
    learning_rate = 0.25
    backup = 'q'
    validate = True
    Plot = MyLearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
    for epsilon in epsilons:       
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n, validate = validate, interval=interval)
        Plot.add_curve_validate(validation_points,learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
    
    policy_annealing = True
    policy_linear_annealing = Linear_anneal(0.3,0.02,80/100)
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n, policy_annealing,
                                                policy_linear_annealing, validate =validate, interval=interval)
    Plot.add_curve_validate(validation_points,learning_curve,label=r'$\epsilon$-greedy, $\epsilon $- annealing')

    policy = 'softmax'
    temps = [0.01,0.1,1.0]
    for temp in temps:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n, validate =validate, interval=interval)
        Plot.add_curve_validate(validation_points,learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))

    policy_annealing = True
    policy_linear_annealing = Linear_anneal(1,0.01,80/100)
    
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n, policy_annealing,
                                                policy_linear_annealing,validate =validate, interval=interval)
    Plot.add_curve_validate(validation_points,learning_curve,label=r'softmax, $ \tau $ - annealing')

    
    policy = 'UCB1'
    Cs = [1,5]
    for C in Cs:     
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n,
                                                validate = validate,interval=interval)
        Plot.add_curve_validate(validation_points,learning_curve,label=r'UCB1, C = {}'.format(C))


    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('exploration_mean_reward_greedy_agend_every500iteration.png')

def a3_on_off_policy(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep):
    n = 5
    temp = 1.0
    epsilon = 0.3
    C = 0.1

    policy = 'egreedy'
    epsilon = 0.1 # set epsilon back to original value 
    learning_rates = [0.02,0.1,0.4]
    backups = ['q','sarsa']
    Plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')
        
    for backup in backups:
        for learning_rate in learning_rates:
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, C, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
        lr_annealing = True
        lr_linear_annealing = Linear_anneal(0.4,0.02,80/100)
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, C, smoothing_window, plot, n, lr_annealing = lr_annealing, lr_linear_annealing = lr_linear_annealing)
        Plot.add_curve(learning_curve,label=r'{}, $\alpha$ - annealing '.format(backup_labels[backup]))
            
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('on_off_policy_training_reward.png')

def a3_on_off_policy_validate(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep):
    n = 5
    temp = 1.0
    epsilon = 0.3
    C = 1
    interval = 500
    validation_points = np.array([interval*(i+1) for i in range(np.int32(n_timesteps/interval))]) 

    policy = 'egreedy'
    epsilon = 0.1 # set epsilon back to original value 
    learning_rates = [0.02,0.1,0.4]
    backups = ['q','sarsa']
    Plot = MyLearningCurvePlot(title = 'Back-up: on-policy versus off-policy')
        
    for backup in backups:
        for learning_rate in learning_rates:
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp,C, smoothing_window, plot, n, validate=True, interval=interval)
            Plot.add_curve_validate(validation_points,learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
        lr_annealing = True
        lr_linear_annealing = Linear_anneal(0.4,0.02,80/100)
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp,C, smoothing_window, plot, n, lr_annealing = lr_annealing, lr_linear_annealing = lr_linear_annealing, validate=True, interval=interval)
        Plot.add_curve_validate(validation_points,learning_curve,label=r'{}, $\alpha$ - annealing '.format(backup_labels[backup]))
            
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('on_off_policy_mean_reward_greedy_agend_every500iteration.png')
    
def a4_depth(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep):
    n = 5
    temp = 1.0
    epsilon = 0.3
    C = 1
    
    policy = 'egreedy'
    epsilon = 0.1 # set epsilon back to original value
    learning_rate = 0.25
    backup = 'nstep'
    ns = [1,3,10,30]
    Plot = LearningCurvePlot(title = 'Back-up: depth')    
    for n in ns:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp,C, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'{}-step Q-learning'.format(n))
    backup = 'mc'
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                          gamma, policy, epsilon, temp,C,  smoothing_window, plot, n)
    Plot.add_curve(learning_curve,label='Monte Carlo')        
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('depth_training_reward.png')


def a4_depth_validate(n_repetitions, smoothing_window, backup_labels, plot, n_timesteps, max_episode_length, gamma, optimal_average_reward_per_timestep):
    n = 5
    temp = 1.0
    epsilon = 0.3
    C=1
    interval = 500
    validation_points = np.array([interval*(i+1) for i in range(np.int32(n_timesteps/interval))])

    policy = 'egreedy'
    epsilon = 0.1 # set epsilon back to original value
    learning_rate = 0.25
    backup = 'nstep'
    ns = [1,3,10,30]
    Plot = MyLearningCurvePlot(title = 'Back-up: depth')    
    for n in ns:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, C, smoothing_window, plot, n, validate=True, interval=interval)
        Plot.add_curve_validate(validation_points,learning_curve,label=r'{}-step Q-learning'.format(n))
    backup = 'mc'
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                          gamma, policy, epsilon, temp, C, smoothing_window, plot, n, validate=True, interval=interval)
    Plot.add_curve_validate(validation_points,learning_curve,label='Monte Carlo')        
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('depth_mean_reward_greedy_agend_every500iteration.png')


class Linear_anneal:
    def __init__(self, start, final, percentage):
        self.start = start
        self.final = final
        self.percentage = percentage
    
    def get_value(self,t,T):
        return linear_anneal(t, T, self.start, self.final, self.percentage)



if __name__ == '__main__':
    experiment()
