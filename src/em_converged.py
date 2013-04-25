import numpy as np

def em_converged(loglik, previous_loglik, threshold, check_increased):
    converged = False
    decrease = False

    if check_increased:
      if loglik - previous_loglik < -.001:
          decrease = False
          converged = False

  

    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.spacing(1))/2
    if (delta_loglik / avg_loglik) < threshold:
        converged =True
        return True