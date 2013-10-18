import numpy as np
from underflow_normalize import normalize,mk_stochastic
from sample import *
from dhmm_em import dhmm_em
O = 3
Q = 2

#"true" parameters
prior0 = normalize(np.random.rand(Q,1))[0]
prior0 =np.array([.5,.5])

transmat0 = mk_stochastic(np.random.rand(Q,Q))
transmat0=np.array([[.3 ,.7],[.7,.3]])
obsmat0 = mk_stochastic(np.random.rand(Q,O))
obsmat0=np.array([[.333,.333,.333],[.333,.333,.333]])
print "True Parameters"
print "-"*80
print "Prior:",prior0
print "Observation",obsmat0
print "Transition:",transmat0



# training data
T = 1
nex =5
[obs,hidden] = sample_dhmm(prior0, transmat0, obsmat0, T, nex)

#print hidden,obs# initial guess of parameters
prior1 = normalize(np.random.rand(Q,1))[0]

prior_1=prior1.flatten(1)
prior_1=prior0
transmat1 = mk_stochastic(np.random.rand(Q,Q))
obsmat1 = mk_stochastic(np.random.rand(Q,O))

obs=np.array([[ 2,1,0,0,2,1,0,2,2,1],[2,0,1,0,0,0,2,2,1,2],[1,2,0,0,2,2,1,0,2,2]])
''',
              [0,0,1,2,0,2,0,2,0,2,1,1],[0,1,2,0,1,1,0,1,2,0]])'''
'''hidden=np.array([[1    1    2    1    2    1    1    2    1    1
][2    2    1    2    1    2    1    2    1    2
]
                 [1    2    2    1    2    2    1    2    1    2
][1    2    1    2    2    1    1    2    1    1
][1    1    1    1    1    1    2    1    2    1
]])'''
transmat1=  np.array([[0.3143, 0.6857],[0.4807, 0.5193]])
obsmat1= np.array([[0.2535, 0.2571, 0.4893],[0.3934,0.4136,0.1930]])
prior_1=np.array([.30,.70])
print "Train initial guess parameters"
print "-"*80
print "Prior:",prior_1
print "Observation",obsmat1 
print "Transition:",transmat1
#transmat1=transmat0
#obsmat1=obsmat0
# prior1=prior0;
# transmat1=transmat0
# obsmat1=obsmat0    

# improve guess of parameters using EM

[LL, prior2, transmat2, obsmat2,nr_iter] = dhmm_em(obs, prior_1, transmat1, obsmat1, 3500,.0000001 );


print "Learnt Parameters"
print "Prior:",prior2
print "Transition:",transmat2
print "Observation:",obsmat2
import matplotlib.pyplot as plt
#plt.plot(LL)
#plt.show()