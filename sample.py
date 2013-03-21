import numpy as np

def sample_discrete(prob,rows=1,columns=1):
        n=len(prob)
        R=np.random.rand(rows,columns)
        M=np.zeros((rows,columns),dtype=np.int)
        cumprob=np.cumsum(prob)
        if n < rows*columns:
            for i in range(0,n-1):
                M = M + (R > cumprob[i])
        else:
            cumprob2 = cumprob[:-1]
            for i in range(0,rows):
                for j in range(0,columns):
                    M[i][j] = sum(R[i][j] > cumprob2)
       
        
        return M
def sample_mc(initial_probability,tranistion_probability,sequence_length,number_of_sequences=1):
    S=np.zeros((number_of_sequences,sequence_length),dtype=np.int)
    for i in range(number_of_sequences):
        S[i][0] = sample_discrete(initial_probability)
        for t in range(1,sequence_length):
            S[i][t] = sample_discrete(transition_probability[S[i][t-1],:])
    return S    
  
def sample_dhmm(initial_probability,transition_probability,observation_matrix,number_of_sequences,sequence_length):
    hidden_samples = sample_mc(initial_probability, transition_probability, sequence_length, number_of_sequences)
    #print hidden_samples
    observed_samples = sample_multinomial(hidden_samples, observation_matrix)
    return [hidden_samples,observed_samples]

def sample_multinomial(observed_state_sequence, observation_matrix):
    #print observed_state_sequence,observed_state_sequence.shape
    print observed_state_sequence,observation_matrix
    Y=np.zeros(shape=np.size(observed_state_sequence))
    #print Y.size,len(Y)
    flat_observation_matrix=observed_state_sequence.flatten(1)
    #print flat_observation_matrix
    for i in range(min(flat_observation_matrix),max(flat_observation_matrix)+1):
        
        index=np.nonzero(flat_observation_matrix==i)
        #print index
        #print "Index:",index,i,obser
        #print "Sample: ",sample_discrete(observation_matrix[i,:], len(index), 1)
        Y[index] = sample_discrete(observation_matrix[i,:], len(index), 1)
        Z=Y.reshape(observed_state_sequence.shape)
    return Z
        


'''Test function
Based on Wikipedia article on HMM
Not we use 0 indexing
start_probability={
    'Rainy': 0.6, 'Sunny': 0.4
    }
transition_probability = {
   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
   }
emission_probability = {
   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
   }
   
'''
    
    
start_probability=np.array([.9,.1])
transition_probability=np.array([[.5,.5],[.5,.5]])
emission_probability=np.array([[.9,.1,.0],[.3,.6,.1]])

sample_discrete(start_probability, 3,10)

print sample_dhmm(start_probability,transition_probability,emission_probability,3,5)


                                 