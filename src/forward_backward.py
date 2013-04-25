
import numpy as np
from underflow_normalize import normalize


def forward_backward(prior,transition_matrix,emission_matrix,observation_vector,scaling,observation_likelihood):
    print "PRIOR OBTAINED IS:",prior
    number_of_hidden_states=len(prior)
    number_of_observations=len(observation_vector)
    shape_alpha=(number_of_hidden_states,number_of_observations)
    print shape_alpha
    alpha=np.zeros(shape_alpha)
    scale=np.ones(number_of_observations)
    xi_summed = np.zeros((number_of_hidden_states,number_of_hidden_states))
    gamma=np.zeros(shape_alpha)
    
    
    '''Forwards'''
    
    '''1.Initialization'''
    t=0
    '''
    first_observation=observation_vector[t]
    print "Forward Backward"
    print prior,emission_matrix[first_observation:,t]
    print prior*emission_matrix[first_observation:,t],np.shape(prior*emission_matrix[first_observation:,t])
    print
    print np.dot(prior,emission_matrix[first_observation:,t])
    alpha[:,t]=np.dot(prior,emission_matrix[first_observation:,t])
    '''
    for i in range(0,number_of_hidden_states):
        print prior[i],prior
        print prior[i]*emission_matrix[i][observation_vector[t]]
        alpha[i][0]=prior[i]*emission_matrix[i][observation_vector[t]]
    
    
    if scaling:
        [alpha[:,0], n] = normalize(alpha[:,0])
        scale[0] = 1/n
    else:
        pass
    
    '''2.Induction'''
    '''Currently Non-vectorized'''
    for t in range(1,number_of_observations):
        for j in range(0,number_of_hidden_states):
            prob_sum=0            
            for i in range(0,number_of_hidden_states):
                prob_sum+=alpha[i][t-1]+transition_matrix[i][j]                
            alpha[j][t]=prob_sum*emission_matrix[j][observation_vector[t]]
        if scaling:
            [alpha[:,t], n] = normalize(alpha[:,t])
            scale[t] = 1/n
        print "SHAPE OF XI :",np.shape(xi_summed)
        print normalize(np.dot(alpha[:,t-1] ,observation_likelihood[:,t].conj().T) * transition_matrix)[0]
        xi_summed = xi_summed + normalize(np.dot(alpha[:,t-1] ,observation_likelihood[:,t].conj().T) * transition_matrix)[0]
    
    '''3.Termination'''
    if scaling:
        loglik=sum(np.log(scale))
    else:
        loglik=np.log(sum(alpha[:,number_of_observations]))

    '''Backwards'''

   
    beta=np.ones(shape_alpha)

    
    
    '''1.Initialization'''
    '''We have already set beta as a sequence of 1's.'''
    
    '''2.Induction'''
    '''Currently Non-vectorized'''
    for t in range(number_of_observations-2,1,-1):
        b = np.dot(beta[:,t+1] ,observation_likelihood[:,t+1])
        for i in range(0,number_of_hidden_states):
            beta_sum=0            
            for j in range(0,number_of_hidden_states):
                beta_sum+=(beta[j][t+1]*transition_matrix[i][j]*emission_matrix[j][observation_vector[t+1]])         
            beta[i][t]=beta_sum
        if scaling:
            [beta[:,t], n] = normalize(beta[:,t]);
            scale[t] = 1/n;
    gamma[:,number_of_observations-1] = normalize(np.dot(alpha[:,number_of_observations-1] ,beta[:,number_of_observations-1]))
    xi_summed  = xi_summed + normalize((transition_matrix * np.dot(alpha[:,t] , b.conj().T)))[0]
    
    return [alpha,beta,gamma,loglik,xi_summed]

    
      
