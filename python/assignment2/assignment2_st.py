import numpy as np
import numpy.random as npr
import math as math
import matplotlib.pyplot as plt
from scipy.special import gammaln
import scipy.stats as ss

# a function for calculating the log of the likelihood for the student-t distribution
def loglikelihood(nu,mu,sigma):
    result = sum(gammaln((nu+1.)/2.) - gammaln(nu/2.) -(1./2.)*np.log(math.pi) - (1./2.)*np.log(nu) - np.log(sigma) - ((nu+1.)/2.)*np.log(1. + (((indData-mu)/sigma)**2.)/nu))
    return result

N=1000                     # number of samples
autoL=100                   # autocoreelation length to get independet samples
indData = np.genfromtxt("data.dat", delimiter=" ")
datL = len(indData)
length=N*autoL
data=np.zeros((length,6))
accept = 0
alpha = 1.

# eigenvectors of fisher matrix, calculated analytically in mathematica
evec = np.zeros((3,3),dtype=float)
evec[0] = [(1./6.)*(65. - 3.*math.pi**2. - math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.)) , 0. , 1.]
evec[1] = [0. , 1. , 0.]
evec[2] = [(1./6.)*(65. - 3.*math.pi**2. + math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.)) , 0. , 1.]


# eigenvalues of fisher matrix, calculated analytically from mathemaica
evalue = [0.,0.,0.]
evalue[2] = (1./72.)*(7. + 3.*math.pi**2. + math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.))
evalue[1] = 2./3.
evalue[0] = (1./72.)*(7. + 3.*math.pi**2. - math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.))


# initial guess for parameters (use the correct ones we know for know)
param = [0,0,0]
newp = [0.,0.,0.]
param[0] = 3.        # nu
param[1] = 1.        # mu
param[2] = 1.        # sigma

#
# loop for mcmc sampling of the distribution
#

oldlike = loglikelihood(param[0],param[1],param[2])
for i in range(0,length):

    # select a new random sample
    j = np.random.choice((0,1,2))
    newp[0] = param[0] + 0.05*npr.normal(0,alpha,1)[0]*(evec[j][0])/math.sqrt(evalue[j]*3.)
    newp[1] = param[1] + 0.05*npr.normal(0,alpha,1)[0]*(evec[j][1])/math.sqrt(evalue[j]*3.)
    newp[2] = param[2] + 0.05*npr.normal(0,alpha,1)[0]*(evec[j][2])/math.sqrt(evalue[j]*3.)
    u = npr.uniform(0,1,1)[0]
    
    if (newp[0]>.1 and newp[0]<10. and newp[1]>-5. and newp[1]<5. and newp[2]>0.1 and newp[2]<10.):

        # calculate the Metropolis-Hastings ratio for the new and old sample
        newlike = loglikelihood(newp[0],newp[1],newp[2])
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
p3 = data[:,3]
C1 =  np.correlate((p1 - np.mean(p1)),(p1 - np.mean(p1)),mode='full')[(length -1):]
C2 =  np.correlate((p2 - np.mean(p2)),(p2 - np.mean(p2)),mode='full')[(length -1):]
C3 =  np.correlate((p3 - np.mean(p3)),(p3 - np.mean(p3)),mode='full')[(length -1):]
cor1=C1/(np.max(C1))
cor2=C2/(np.max(C2))
cor3=C3/(np.max(C3))
autol = [0,0,0]
autol[0] = np.where(cor1<0.01)[0]
autol[1] = np.where(cor2<0.01)[0]
autol[2] = np.where(cor3<0.01)[0]

print autol[0][0]
print autol[1][0]
print autol[2][0]

plt.figure(1)
plt.plot(p1,p3,'.')
plt.savefig('nu-sigma.pdf')
plt.figure(2)
plt.plot(p1,p2,'.')
plt.savefig('nu-mu.pdf')
plt.figure(3)
plt.plot(p2,p3,'.')
plt.savefig('mu-sigma.pdf')








