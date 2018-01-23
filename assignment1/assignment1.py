import numpy as np
import numpy.random as npr
import math as math
import matplotlib.pyplot as plt
from scipy.special import gammaln

# a function for calculating the log of the likelihood for the student-t distribution
def loglikelihood(xx):
    nu=3.
    mu=1.
    sigma=1.
    result = gammaln((nu+1.)/2.) - gammaln(nu/2.) -(1./2.)*np.log(math.pi*nu) - np.log(sigma) - ((nu+1.)/2.)*np.log(1. + ((xx-mu)/sigma)**2./nu)
    return result

N=10000                     # number of samples
autoL=20                   # autocoreelation length to get independet samples
length=N*autoL
x0=npr.normal(0,2,1)[0]     # initial sample
alpha = 10.                 # standard deviation of proposal distribution
data=np.zeros((length,2))        # initializing the output array
output='alpha_xxx.dat'           # name of output fiel

#
# loop for mcmc sampling of the distribution
#
for i in range(0,length):

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
samples=data[:,0]

#
# Post processing
#

# getting the acceptance fraction
acceptance=100*sum(data[:,1])/length
print acceptance

# finding the atucorrelation length

p0 = samples
C0 =  np.correlate((p0 - np.mean(p0)),(p0 - np.mean(p0)),mode='full')[(length -1):]
cor=C0/(np.max(C0))
autol = np.where(cor<0.01)
print autol[0][0]

# Plotting

import scipy.stats as ss

plot_title = 'xxx.pdf'
plt.figure(1)
plt.plot(samples)
#title = 'Student t MCMC Sampler Chain, $\\alpha$ = '+alpha+', \nAcceptance Rate: '+str(acceptance_rate)+'%'
#plt.title(title)
plt.savefig(plot_title)

x = np.linspace(min(samples),max(samples),100)
y = ss.t.pdf(x,3.,1.,1.)
n = len(samples)
n_bins = 100
err = 1./np.sqrt(n*y)
nSig = 3.

plt.figure(2)
plt.hist(data[:,0],bins=n_bins,histtype='step',normed=True)
plt.plot(x,y,'k-.',linewidth=2)
plt.fill_between(x,y*(1.-nSig*err),y*(1.+nSig*err),alpha=0.3,facecolor='red')
plt.yscale('log', nonposy='clip')
#title = 'Student t MCMC Sampler (2$\\sigma$) $\\alpha$ ='+alpha
#plt.title(title)
plt.savefig('histogram.pdf')

indData=np.zeros((N,1))
for i in range(0,length):
    if (i%autoL==0):
        indData[i/autoL]=data[i,0]

np.savetxt('data.dat',indData,fmt='%f')









