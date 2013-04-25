'''Backward algorithm
It solves the first portion of the first problem as per Rabiner's classic paper on HMM
It tries to compute Probability(Observation Sequence| Lambda=(Pi,A,B))
'''

import numpy as np
from underflow_normalize import normalize


def backward(prior,transition_matrix,emission_matrix,observation_vector,scaling=True):
    number_of_hidden_states=len(prior)
    number_of_observations=len(observation_vector)
    shape_alpha=(number_of_hidden_states,number_of_observations)
    beta=np.ones(shape_alpha)
    scale=np.ones(number_of_observations)
    
    
    '''1.Initialization'''
    '''We have already set beta as a sequence of 1's.'''
    
    '''2.Induction'''
    '''Currently Non-vectorized'''
    for t in range(number_of_observations-1,1,-1):
        for i in range(0,number_of_hidden_states):
            beta_sum=0            
            for j in range(0,number_of_hidden_states):
                beta_sum+=beta[i][t-1]+transition_matrix[i][j]                
            beta[i][t]=beta_sum*emission_matrix[j][observation_vector[t]]
        if scaling:
            [beta[:,t], n] = normalize(beta[:,t]);
            scale[t] = 1/n;
    
   
   
