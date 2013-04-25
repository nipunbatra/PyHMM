import numpy as np
from evaluate_pdf_cond_multinomial import evaluate_pdf_cond_multinomial

'''Compute Sufficient Statistics
1. Prior
2. Transition Matrix
3. Emission Matrix
4. Observation matrix
'''
def compute_ess_dmm(prior,transition_matrix,emission_matrix,observation_vector, dirichlet):
    (S,O)=np.size(emission_matrix)
    exp_num_tran=np.zeros((S,S))
    exp_num_visits1=np.zeros((S,1))
    exp_num_visitsT=np.zeros((S,1))
    exp_num_emit = dirichlet*np.ones((S,O))
    loglik = 0
    
    #Number of observations
    T=len(observation_vector)
    obslik = evaluate_pdf_cond_multinomial(observation_vector, emission_matrix)