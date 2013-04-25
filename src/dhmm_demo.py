import numpy as np
from underflow_normalize import normalize,mk_stochastic
from sample import *
from dhmm_em import dhmm_em
O = 3
Q = 2

#"true" parameters
prior0 = normalize(np.random.rand(Q,1))[0]
print "Prior:",prior0
transmat0 = mk_stochastic(np.random.rand(Q,Q))
print "Transition:",transmat0
obsmat0 = mk_stochastic(np.random.rand(Q,O))
print "Observation",obsmat0


# training data
T = 5
nex = 10
[obs,hidden] = sample_dhmm(prior0, transmat0, obsmat0, T, nex)

print hidden,obs# initial guess of parameters
prior1 = normalize(np.random.rand(Q,1))[0]
prior_1=prior1.flatten(1)
print "SENDING PRIOR",prior1,prior_1
transmat1 = mk_stochastic(np.random.rand(Q,Q))
obsmat1 = mk_stochastic(np.random.rand(Q,O))

# prior1=prior0;
# transmat1=transmat0
# obsmat1=obsmat0

# improve guess of parameters using EM
[LL, prior2, transmat2, obsmat2] = dhmm_em(obs[0],hidden[0], prior1, transmat1, obsmat1, 5000,.0001 );
