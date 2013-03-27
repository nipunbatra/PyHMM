'''Forward algorithm
It solves the first portion of the first problem as per Rabiner's classic paper on HMM
It tries to compute Probability(Observation Sequence| Lambda=(Pi,A,B))
'''

import numpy as np
from underflow_normalize import normalize


def forward(prior,transition_matrix,emission_matrix,observation_vector,scaling=True):
    number_of_hidden_states=len(prior)
    number_of_observations=len(observation_vector)
    shape_alpha=(number_of_hidden_states,number_of_observations)
    alpha=np.zeros(shape_alpha)
    scale=np.ones(number_of_observations)
    
    
    '''1.Initialization'''
    t=0
    first_observation=observation_vector[t]
    alpha[:,t]=prior*emission_matrix[first_observation:,t]
    if scaling:
        [alpha[:,0], n] = normalize(alpha[:,0])
        scale[0] = 1/n;
    
    '''2.Induction'''
    '''Currently Non-vectorized'''
    for t in range(1,number_of_observations):
        for j in range(0,number_of_hidden_states):
            prob_sum=0            
            for i in range(0,number_of_hidden_states):
                prob_sum+=alpha[i][t-1]+transition_matrix[i][j]                
            alpha[j][t]=prob_sum*emission_matrix[j][observation_vector[t]]
        if scaling:
            [alpha[:,t], n] = normalize(alpha[:,t]);
            scale[t] = 1/n;
    
    '''3.Termination'''
    if scaling:
        loglik=sum(np.log(scale))
    else:
        loglik=np.log(sum(alpha[:,number_of_observations]))
    return alpha,loglik
   
