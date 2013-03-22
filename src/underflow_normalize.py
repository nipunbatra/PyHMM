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
        
'''      
c=np.array([[.7,.3, .3],[.4,.6,.4]])
[M,s]=normalize(c, 2)
print M,s
'''