import numpy as np

def normalize(A,dim=None):
    if dim is None:
        z=sum(A.flatten(1))
        s=z+(z==0.0)
        M=A/s
    elif dim==1:
        z=np.sum(A)
        s=z+(z==0.0)
        M=A/s
    else:
        
        z=np.sum(A,axis=dim-1)
        print z
        
        s = z + (z==0)
        s=s[:, np.newaxis]
        M=A/s
    return [M,s]


def mk_stochastic(T):
    if (np.ndim(T)==2) and (T.shape[0]==1 or T.shape[1]==1): # isvector
        [T,Z] = normalise(T);
    else:
        number_of_rows=np.shape(T)[0]
        for row in range(0,number_of_rows):
            sum_row=np.sum(T[row,:])
            T[row,:]=T[row,:]/sum_row
    return T
        
'''      
c=np.array([[.7,.3, .3],[.4,.6,.4]])
[M,s]=normalize(c, 2)
print M,s
'''