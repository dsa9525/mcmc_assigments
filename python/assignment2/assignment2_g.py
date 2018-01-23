import numpy as np
import numpy.random as npr
import math as math
import matplotlib.pyplot as plt
from scipy.special import gammaln
import scipy.stats as ss

# a function for calculating the log of the likelihood for the student-t distribution
def loglikelihood(alpha,beta):
    result = sum(-0.5*np.log(math.pi) - np.log(alpha) - (indData-beta)*(indData-beta)/2/alpha/alpha)
    return result

N=1000                     # number of samples
autoL=100                   # autocoreelation length to get independet samples
indData = np.genfromtxt("data.dat", delimiter=" ")
datL = len(indData)
length=N*autoL
data=np.zeros((length,6))
accept = 0
alpha = 10.

# eigenvectors of fisher matrix, calculated analytically in mathematica
evec = np.zeros((2,2),dtype=float)
evec[0] = [-1. , 0.]
evec[1] = [0. , -1.]


# eigenvalues of fisher matrix, calculated analytically from mathemaica
evalue = [0.,0.]
evalue[0] = 2.66682
evalue[1] = 1.33341



# initial guess for parameters (use the correct ones we know for know)
param = [0,0,0]
newp = [0.,0.,0.]
param[0] = 0.866    # alpha, initial guess with alpha = sigma*sqrt(nu/(nu+1.))
param[1] = 1.       # beta
param[2] = 0.       # null position

#
# loop for mcmc sampling of the distribution
#

oldlike = loglikelihood(param[0],param[1])
for i in range(0,length):

    # select a new random sample
    j = np.random.choice((0,1))
    newp[0] = param[0] + npr.normal(0,alpha,1)[0]*(evec[j][0])/math.sqrt(evalue[j])
    newp[1] = param[1] + npr.normal(0,alpha,1)[0]*(evec[j][1])/math.sqrt(evalue[j])
    newp[2] = param[2]
    u = npr.uniform(0,1,1)[0]
    
    if (newp[0]>.1 and newp[0]<10. and newp[1]>-5. and newp[1]<5.):

        # calculate the Metropolis-Hastings ratio for the new and old sample
        newlike = loglikelihood(newp[0],newp[1])
        MH_ratio = newlike - oldlike

        # determine whether or not to keep the new sample
        if (np.log(u)<MH_ratio and MH_ratio > -np.inf):
            param[0] = newp[0]
            param[1] = newp[1]
            param[2] = newp[2]
            likehood = newlike
            oldlike=newlike
            accept=1
        else:
            likehood = oldlike
            accept=0

    else:
        accpet = 0
        likehood = oldlike
        oldlike=newlike

    # Output the sample used and whether or not it was accpeted
    data[i,1]=param[0]
    data[i,2]=param[1]
    data[i,3]=param[2]
    data[i,4]=likehood
    data[i,5]=accept

np.savetxt('test.dat',data,fmt='% i %f %f %f %f %i',delimiter='\t')

#
# Post processing
#

# getting the acceptance fraction
acceptance=100*sum(data[:,5])/length
print acceptance

# finding the atucorrelation length

p1 = data[:,1]
p2 = data[:,2]
C1 =  np.correlate((p1 - np.mean(p1)),(p1 - np.mean(p1)),mode='full')[(length -1):]
C2 =  np.correlate((p2 - np.mean(p2)),(p2 - np.mean(p2)),mode='full')[(length -1):]
cor1=C1/(np.max(C1))
cor2=C2/(np.max(C2))
autol = [0,0]
autol[0] = np.where(cor1<0.01)[0]
autol[1] = np.where(cor2<0.01)[0]

print autol[0][0]
print autol[1][0]

plt.figure(1)
plt.plot(p1,p2,'.')
plt.savefig('alpha-beta.pdf')









