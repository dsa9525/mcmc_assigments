import numpy as np
import numpy.random as npr
import math as math
import matplotlib.pyplot as plt
from scipy.special import gammaln

# a function for calculating the log of the likelihood for the student-t distribution
def loglikelihood(xx):
    nu=3
    mu=2
    sigma=1
    result = gammaln((nu+1)/2) - gammaln(nu/2) -(1/2)*np.log(math.pi*nu) - np.log(sigma) - ((nu+1)/2)*np.log(1 + ((xx-mu)/sigma)**2/nu)
    return result

N=1000                      # number of samples
x0=npr.normal(0,2,1)[0]     # initial sample
alpha = 0.1                   # standard deviation of proposal distribution
data=np.zeros((N,2))        # initializing the output array
output='alpha_0.1.dat'           # name of output fiel

#
# loop for mcmc sampling of the distribution
#
for i in range(0,N):

    # select a new random sample
    x1 = x0 + npr.normal(0,alpha,1)[0]
    u = npr.uniform(0,1,1)[0]

    # calculate the Metropolis-Hastings ratio for the new and old sample
    MH_ratio = math.exp(loglikelihood(x1) - loglikelihood(x0))

    # determine whether or not to keep the new sample
    if (u<MH_ratio):
        x0=x1
        accept=1
    else:
        accept=0

    # Output the sample used and whether or not it was accpeted
    data[i,0]=x0
    data[i,1]=accept

np.savetxt(output,data,fmt='%f %i',delimiter='\t')
