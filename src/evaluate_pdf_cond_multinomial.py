
import numpy as np

def evaluate_pdf_cond_multinomial(data, obsmat):
    (Q,O) = np.shape(obsmat)
    T = len(data)
    B = np.zeros((Q,T))

    for t in range(0,T):
        B[:,t] = obsmat[:, data[t]]


    return B    
