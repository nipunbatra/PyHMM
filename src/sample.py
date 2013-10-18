import numpy as np
from pyx.style import linewidth

def sample_discrete(prob,rows=None,columns=None):
        n=len(prob)
        if rows is None:
            rows=1
        if columns is None:
            columns=1
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
def sample_mc(initial_probability,transition_probability,sequence_length,number_of_sequences=1):
    S=np.zeros((number_of_sequences,sequence_length),dtype=np.int)
    for i in range(number_of_sequences):
        S[i][0] = sample_discrete(initial_probability)
        for t in range(1,sequence_length):
            S[i][t] = sample_discrete(transition_probability[S[i][t-1]-1,:])
    return S    
  
def sample_dhmm(initial_probability,transition_probability,observation_matrix,number_of_sequences,sequence_length):
    hidden_samples = sample_mc(initial_probability, transition_probability, sequence_length, number_of_sequences)
    #print hidden_samples
    observed_samples = sample_multinomial(hidden_samples, observation_matrix)
    return [observed_samples,hidden_samples]

def sample_multinomial(observed_state_sequence, observation_matrix):
    #print observed_state_sequence,observed_state_sequence.shape
    #print observed_state_sequence,observation_matrix
    Y=np.zeros(shape=np.size(observed_state_sequence),dtype=np.int)
    #print Y.size,len(Y)
    #raw_input()
    flat_observation_matrix=observed_state_sequence.flatten(1)
    #raw_input()
    #print flat_observation_matrix
    for i in range(min(flat_observation_matrix),max(flat_observation_matrix)+1):
       
        index=np.where(flat_observation_matrix==i)[0]
        
        #print index
        #print "Index:",index,i,obser
        #print "Sample: ",sample_discrete(observation_matrix[i,:], len(index), 1)
        Y[index] = sample_discrete(observation_matrix[i,:], len(index), 1)
        Z=Y.reshape(observed_state_sequence.shape)
    return Z

def apply_observation_map(x):
    if x==0:
        return "Walk"
    elif x==1:
        return "Shop"
    else:
        return "Clean"
    
def apply_hidden_map(x):
    if x==0:
        return "Rainy"
    else:
        return "Sunny"
         


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
    
start_probability=np.array([.5,.5])
transition_probability=np.array([[.7,.3],[.4,.6]])
emission_probability=np.array([[.1,.4,.5],[.6,.3,.1]])

sample_discrete(start_probability, 1,20)

[observed,hidden]=sample_dhmm(start_probability,transition_probability,emission_probability,2,100)
#print hidden
#print observed
'''
for i in range(len(observed)):
    print hidden[i]," ->",observed[i]
hidden_names=np.vectorize(apply_hidden_map)(hidden)
observed_names=np.vectorize(apply_observation_map)(observed)

import matplotlib.pyplot as plt
a=np.ones(100)
indx1=np.where(hidden[0]==0)[0]
print indx1,hidden[0][indx1]
indx2=np.where(hidden[0]==1)[0]
#plt.fill_between(hidden[0][indx1], np.ones(len(indx1)), np.ones(len(indx1)))
#f
import matplotlib

x=indx1
print hidden[0]==0,sum(hidden[0]==0)

plt.plot(hidden[0],linewidth=2)
plt.ylim((-0.02,1.02))
plt.figtext('')
plt.show()


def group(L):
    first = last = L[0]
    for n in L[1:]:
        if n - 1 == last: # Part of the group, bump the end
            last = n
        else: # Not part of the group, yield current group and start a new
            yield first, last
            first = last = n
    yield first, last # Yield the last group
#raw_input()
#plt.fill_between(range(100), 0, 1,where=(hidden[0]==0))
#plt.fill_between(range(100), 0, 1,where=(hidden[0]==1),color='g')



#for i in range(0,len(indx1)-1):
    #plt.axvspan(indx1[i],indx1[i+1], facecolor='g', alpha=0.5)


    #plt.fill_between([indx1[i]], 0, 1,color='g')
#plt.show()

#plt.plot(observed[0])
#plt.axes().patches.Rectangle((-200,-100), 400, 200, color='yellow')

indx1_groups=list(group(indx1))
indx1_tuples=[(x,y-x+1) for (x,y) in indx1_groups]

indx2_groups=list(group(indx2))
indx2_tuples=[(x,y-x+1) for (x,y) in indx2_groups]

fig = plt.figure()
ax = fig.add_subplot(111)
rect1=[]
rect2=[]
print indx1
tuple_broken=[(indx1[i],indx1[i+1]-indx1[i]) for i in range(len(indx1)-1)]
print tuple_broken
for i in range(len(indx1)-1):
    ax.broken_barh(indx1_tuples,(0,1),facecolor='r')
for i in range(len(indx2)-1):
    ax.broken_barh(indx2_tuples,(0,1),facecolor='g')
print "Index 1",indx1_tuples
print "Index 2",indx2_tuples

plt.show()
    #rect1.append(matplotlib.patches.Rectangle((indx1[i],0), indx1[i+1]-indx1[i], 1, color='yellow'))
    #ax.add_patch(rect1[i])
#for i in range(len(indx2)-1):
    #rect2.append(matplotlib.patches.Rectangle((indx2[i],0), indx2[i+1]-indx1[i], 1, color='green'))
    #ax.add_patch(rect2[i])
#plt.xlim((0,100))
#plt.show()
#plt.plot(observed[0])

#plt.plot(hidden[0])
#plt.show()


#print "Rainy: ",np.sum(hidden_names=='Rainy')
'''
