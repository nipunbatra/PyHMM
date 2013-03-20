import numpy as np

def sample_discrete(prob,r=1,c=1):
        n=len(prob)
        R=np.random.rand(r,c)
        M=np.ones((r,c))
        cumprob=np.cumsum(prob)
        print cumprob
        if n < r*c:
            for i in range(0,n-1):
                M = M + (R > cumprob[i])
        else:
            cumprob2 = cumprob[:-1]
            for i in range(0,r):
                for j in range(0,c):
                    M[i][j] = sum(R[i][j] > cumprob2)+1
        return M
  



print sample_discrete(np.array([0.2,0.8]),2,10)