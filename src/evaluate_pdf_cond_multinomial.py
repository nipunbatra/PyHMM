
import numpy as np

def evaluate_pdf_cond_multinomial(data, obsmat):
    '''% EVAL_PDF_COND_MULTINOMIAL Evaluate pdf of conditional multinomial 
% function B = eval_pdf_cond_multinomial(data, obsmat)
%
% Notation: Y = observation (O values), Q = conditioning variable (K values)
%
% Inputs:
% data(t) = t'th observation - must be an integer in {1,2,...,K}: cannot be 0!
% obsmat(i,o) = Pr(Y(t)=o | Q(t)=i)
%
% Output:
% B(i,t) = Pr(y(t) | Q(t)=i)

data array([1, 1, 2, 1, 1, 3, 2, 3, 2, 3])
array([[ 0.0284 ,  0.315  ,  0.06565],
       [ 0.3154 ,  0.5503 ,  0.1343 ]])

Output: array([[ 0.0284 ,  0.0284 ,  0.315  ,  0.0284 ,  0.0284 ,  0.06565,
         0.315  ,  0.06565,  0.315  ,  0.06565],
       [ 0.3154 ,  0.3154 ,  0.5503 ,  0.3154 ,  0.3154 ,  0.1343 ,
         0.5503 ,  0.1343 ,  0.5503 ,  0.1343 ]])
         
'''


    (Q,O) = np.shape(obsmat)
    
    T = len(data)
    B = np.zeros((Q,T))

    for t in range(0,T):
        B[:,t] = obsmat[:, data[t]    ]


    return B    

