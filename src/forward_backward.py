
import numpy as np
from underflow_normalize import normalize


def forward_backward(prior,transition_matrix,emission_matrix,observation_vector,scaling,obslik):
    number_of_hidden_states=len(prior)
    number_of_observations=len(observation_vector)
    shape_alpha=(number_of_hidden_states,number_of_observations)
    alpha=np.zeros(shape_alpha)
    scale=np.ones(number_of_observations)
    xi = np.zeros((number_of_observations,number_of_hidden_states,number_of_hidden_states)) 
    gamma=np.zeros(shape_alpha)
    gamma2=np.zeros(shape_alpha)

    xi_summed=np.zeros((number_of_hidden_states,number_of_hidden_states))
    
   
    
    
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
        alpha[i][0]=prior[i]*emission_matrix[i][observation_vector[0]]
    
    #print "Prior_i",prior[i]
    #print "Obs",observation_vector[0]
    #print emission_matrix[i][observation_vector[0]]
    
          
    if scaling:
        [alpha[:,0], n] = normalize(alpha[:,0])
        
        scale[0] = n
    else:
        pass
    
    '''2.Induction'''
    '''Currently Non-vectorized'''
    for t in range(1,number_of_observations):
        for j in range(0,number_of_hidden_states):
            prob_sum=0            
            for i in range(0,number_of_hidden_states):
                prob_sum+=alpha[i][t-1]*transition_matrix[i][j]                
            alpha[j][t]=prob_sum*emission_matrix[j][observation_vector[t]]
        if scaling:
            [alpha[:,t], n] = normalize(alpha[:,t])
            scale[t] = n
      
    '''3.Termination'''
    if scaling:
        #print scale,"SCALE"
        loglik=sum(np.log(scale))
    else:
        loglik=np.log(sum(alpha[:,number_of_observations-1]))
    
   
    '''Backwards'''
    beta=np.ones(shape_alpha)
    gamma[:,number_of_observations-1] = normalize(alpha[:,number_of_observations-1] * beta[:,number_of_observations-1])[0]
   
    '''1.Initialization'''
    '''We have already set beta as a sequence of 1's.'''
    
    '''2.Induction'''
    '''Currently Non-vectorized'''
    for t in range(number_of_observations-2,-1,-1):
        b = beta[:,t+1] * obslik[:,t+1]
       
       
        for i in range(0,number_of_hidden_states):
            beta_sum=0            
            for j in range(0,number_of_hidden_states):
                beta_sum+=(beta[j][t+1]*transition_matrix[i][j]*emission_matrix[j][observation_vector[t+1]])         
            beta[i][t]=beta_sum
        if scaling:
            [beta[:,t], n] = normalize(beta[:,t])
            scale[t] = n
        gamma[:,t] = normalize(alpha[:,t] * beta[:,t])[0]
       
        a=alpha[:,t].reshape(number_of_hidden_states,1)
        
        #print transition_matrix
        #print b
        #print obslik[:,t+1],"OBSLIK"
        #print np.dot(a, b.conj().T[np.newaxis])
        
        #print np.mat(alpha[:,t])*np.mat(b.conj().T)
        
        xi_summed  = xi_summed + normalize((transition_matrix * np.dot(a, b.conj().T[np.newaxis])))[0]
    
    '''Computing xi'''
    
    for t in range(number_of_observations-1):
        denom=0.0
        for i in range(0,number_of_hidden_states):
            for j in range(0,number_of_hidden_states):
                denom+=alpha[i][t]*transition_matrix[i][j]*emission_matrix[j][observation_vector[t+1]]*beta[j][t+1]
        for i in range(0,number_of_hidden_states):
            for j in range(0,number_of_hidden_states):
                numer=alpha[i][t]*transition_matrix[i][j]*emission_matrix[j][observation_vector[t+1]]*beta[j][t+1]       
                xi[t][i][j]=numer/denom
    

    '''Computing xi_summed
    for t in range(number_of_observations-1):
        for i in range(number_of_hidden_states):
            for j in range(number_of_hidden_states):
                xi_summed[i][j]+=xi[t][i][j]
    
    for t in xrange(number_of_observations):
        for i in xrange(number_of_hidden_states):
            gamma2[i][t] = sum(xi[t][i])'''
 
    '''CHECKING'''
    #print "Asserting"
    for t in range(number_of_observations):
        for i in range(0,number_of_hidden_states):
            p=0.0
            for j in range(number_of_hidden_states):
                p+=xi[t][i][j]
            #print p/gamma[i][t]
     
   
            #assert(p==gamma[t][i])
    return [alpha,beta,gamma,xi,xi_summed,loglik]

    
      
