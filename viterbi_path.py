import numpy as np
from underflow_normalize import normalize

def path(prior,transition_matrix,emission_matrix,observation_vector,scaling=True):
    number_of_hidden_states=len(start_probability)
    number_of_observations=len(observation_vector)
    shape_delta=(number_of_hidden_states,number_of_observations)
    shape_psi=shape_delta
    delta=np.zeros(shape_delta)
    psi=np.zeros(shape_psi,dtype=np.int)
    scale=np.ones(number_of_observations)
    optimum_path=np.zeros(number_of_observations,dtype=np.int)
    
    
    '''1.Initialization'''
    first_observation=observation_vector[0]
    delta[:,0]=prior*emission_matrix[first_observation:,0]
    psi[:,0]=0
    if scaling:
        [delta[:,0], n] = normalize(delta[:,0])
        scale[0] = 1/n;
    

    '''2.Recursion'''
    '''Currently non vectorized'''
    for t in range(1,number_of_observations):
        for j in range(0,number_of_hidden_states):
            p=0            
            for i in range(0,number_of_hidden_states):
                p=delta[i][t-1]*transition_matrix[i][j]*emission_matrix[j][observation_vector[t]]                
                if p>delta[j][t]:
                    delta[j][t]=p  
                    psi[j][t]=i 
        if scaling:
            [delta[:,t], n] = normalize(delta[:,t]);
            scale[t] = 1/n;
    
    '''3.Termination'''
    p_star=max(delta[:,number_of_observations-1])
    optimum_path[number_of_observations-1]=np.argmax(delta[:,number_of_observations-1])
    
    '''4.Path Backtracking'''
    for t in range(number_of_observations-2,-1,-1):
        optimum_path[t]=psi[optimum_path[t+1]][t]
    
    return optimum_path,delta
    
  
start_probability=np.array([.6,.4])
transition_probability=np.array([[.7,.3],[.4,.6]])
emission_probability=np.array([[.1,.4,.5],[.6,.3,.1]])
#observed_sequence=np.array([0,1,1,1])
observed_sequence=np.random.randint(2,size=50000)
print path(start_probability,transition_probability,emission_probability,observed_sequence)
