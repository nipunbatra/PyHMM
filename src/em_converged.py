import numpy as np

def em_converged(loglik, previous_loglik, threshold, check_increased):
    converged = False
    decrease = False
    if check_increased:
      if loglik - previous_loglik < -.000001:
          decrease = True
          converged = False
          return [converged,decrease]
      
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.spacing(1))/2
    if (delta_loglik / avg_loglik) < threshold:
        converged =True
    return [converged,decrease]