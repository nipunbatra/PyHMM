import numpy as np
from evaluate_pdf_cond_multinomial import evaluate_pdf_cond_multinomial
from forward_backward import forward_backward
from underflow_normalize import normalize, mk_stochastic
from em_converged import em_converged

'''Compute Sufficient Statistics
1. Prior
2. Transition Matrix
3. Emission Matrix
4. Observation matrix
'''
def compute_ess_dhmm(observation_vector,prior,transition_matrix,emission_matrix, dirichlet):
    #print "__________________________________"
    #print "Observation:",observation_vector,"\n","Hidden:",hidden,"\n","Prior",prior,"\n","Transition",transition_matrix,"\n","Emission",emission_matrix
    (S,O)=np.shape(emission_matrix)
    exp_num_trans=np.zeros((S,S))
    exp_num_visits1=np.zeros((1,S)).flatten(1)
    exp_num_visitsT=np.zeros((1,S)).flatten(1)
    exp_num_emit = dirichlet*np.ones((S,O))
    loglik = 0
    num_sequences=len(observation_vector)
    for i in range(num_sequences):
        observation_i=observation_vector[i]
        #Number of observations
        T=len(observation_i)
        obslik = evaluate_pdf_cond_multinomial(observation_i, emission_matrix)
        #E
        [alpha, beta, gamma, xi,xi_summed,current_ll] = forward_backward(prior, transition_matrix, emission_matrix,observation_i,True,obslik)
        loglik = loglik + current_ll
 
        exp_num_trans = exp_num_trans + xi_summed
    
        #print "EXPETCED VISISTS 1 BEFORE",exp_num_visits1
        exp_num_visits1 += gamma[:,0]
        #print "EXPETCED VISISTS 1 AFTER",exp_num_visits1
        exp_num_visitsT = exp_num_visitsT + gamma[:,T-1]
        print "Visits 1",exp_num_visits1
        print "Visits T",exp_num_visitsT
        print "EXP NUM TRANS",exp_num_trans
        for t in range(0,T):
            o = observation_i[t]
            exp_num_emit[:,o] = exp_num_emit[:,o] + gamma[:,t]
        #print "Log Likelihood:",loglik
        #print "Exp Num Transitions",exp_num_trans
        #print "Exp Num Visits 1",exp_num_visits1
        #print "Exp Num Visits T",exp_num_visitsT
        print "Exp Num Emit",exp_num_emit
        print "Xi Summed",xi_summed
        print "Sequence",np.array(observation_i)+1
        print "Alpha",alpha
        print "Beta",beta
        #raw_input('After iteration%s'%i)
        
        new_pi=normalize(gamma[:,0])[0]
        new_A=xi_summed/np.sum(gamma,1)
        (number_of_hidden_states,number_of_observation_states)=np.shape(emission_matrix)
        B_new = np.zeros(np.shape(emission_matrix))
        for j in xrange(number_of_hidden_states):
            for k in xrange(number_of_observation_states):
                numer = 0.0
                denom = 0.0
                for t in xrange(T):
                    if observation_i[t] == k:
                        numer += gamma[j][t]
                    denom += gamma[j][t]
                    B_new[j][k] = numer/denom
            
        
        
        print "******************************"
        #print "New Pi:",new_pi
        #print "New A:",new_A
        #print "After normalizing",mk_stochastic(new_A)
    return [loglik,exp_num_visits1,exp_num_visitsT,exp_num_trans,exp_num_emit]
    #return [loglik, new_pi,new_A,B_new]


def dhmm_em(observation_i,prior,transition_matrix,emission_matrix,max_iter,thresh):
    
    previous_loglik = -np.inf
    loglik = 0
    converged = False
    num_iter = 1
    LL = []
    
    while (num_iter <= max_iter) :
         
         #E step
         [loglik,exp_num_visits1,exp_num_visitsT,exp_num_trans,exp_num_emit]=compute_ess_dhmm(observation_i,prior,transition_matrix,emission_matrix, 0)
         #[loglik, prior,A,B] = compute_ess_dhmm(observation_i,hidden,prior,transition_matrix,emission_matrix, 0)
         print "-----------ITERATION "+str(num_iter)+"----------"
         #M Step
         #prior = prior
         prior=normalize(exp_num_visits1)[0]
         #print "AFTER M STEP:",prior
         transition_matrix = mk_stochastic(exp_num_trans)
         
         #emission_matrix = mk_stochastic(B)
         emission_matrix=mk_stochastic(exp_num_emit)
         print "PRIOR:",prior
         print "Transition",transition_matrix
         print "Observation",emission_matrix
         print loglik
        
         num_iter =  num_iter + 1
         [converged,decrease] = em_converged(loglik, previous_loglik, thresh,False)
         previous_loglik = loglik
         LL.append(loglik)
         #print "Log Likelihood:",LL
    return [LL, prior, transition_matrix, emission_matrix, num_iter]

    
   