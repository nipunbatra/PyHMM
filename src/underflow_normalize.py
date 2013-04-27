import numpy as np

def normalize(A,dim=None):
    '''normalize: Make the entries of a multidimensional array sum to 1
    Inputs
    ------
    A: Array of type floating point
    dim: Dimension on which to normalize
    
    Outputs
    -------
    M: Normalized Array
    Z: Normalizing Constant
    
    Examples
    --------
    
    Example 1: Normalizing a vector without specifying dimension
    [normalized_array, normalizing_constant]=normalize(np.array([1,2,3]))
    
    Output
    normalized_array=array([ 0.16666667,  0.33333333,  0.5])
    normalizing_constant=6
    
    Example 2: Normalizing a 2-d matrix without specifying dimension
    [normalized_array, normalizing_constant]=normalize(np.array([[1,2,3],[2,3,4]]))
    
    Output
    normalized_array=array([[ 0.06666667,  0.13333333,  0.2       ],
       [ 0.13333333,  0.2       ,  0.26666667]])
    normalizing_constant=15
    
    Example 3: Normalizing a 2-d matrix across the 'row' dimension (2nd dim)
    [normalized_array, normalizing_constant]=normalize(np.array([[1,2,3],[2,3,4]]),dim=2)
    
    Output
    normalized_array=array([[ 0.16666667,  0.33333333,  0.5       ],
       [ 0.22222222,  0.33333333,  0.44444444]])
    normalized_constant=array([[ 6.],[ 9.]])
    
    NB: While using this function make sure that if you do not require the normalizing
    constant you take the first element of the output

    '''
    if dim is None:
        z=sum(A.flatten(1))
        s=1.0*(z+(z==0.0))
        M=A/s
    elif dim==1:
        z=np.sum(A)
        s=1.0*(z+(z==0.0))
        M=A/s
    else:        
        z=np.sum(A,axis=dim-1)
        s = 1.0*(z + (z==0.0))
        s=s[:, np.newaxis]
        M=A/s
    return [M,s]


def mk_stochastic(T):
    print "Dimensions:",np.ndim(T)
    print "Shape:",np.shape(T)
    if (np.ndim(T)==2) and (T.shape[0]==1 or T.shape[1]==1): # isvector
        [T,Z] = normalise(T);
    else:
        out=np.zeros((np.shape(T)))
        number_of_rows=np.shape(T)[0]
        for row in range(0,number_of_rows):
            
            sum_row=1.0*np.sum(T[row,:])
            
            a=T[row,:]
            b=a/sum_row
            
            out[row,:]=b
            
    
    return out
        
'''      
c=np.array([[.7,.3, .3],[.4,.6,.4]])
[M,s]=normalize(c, 2)
print M,s
'''