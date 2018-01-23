import numpy as np
import numpy.random as npr
import math as math
import matplotlib.pyplot as plt
from scipy.special import gammaln
import scipy.stats as ss

# a function for calculating the log of the likelihood for the student-t distribution
def loglikelihoodS(nu,mu,sigma):
    result = sum(gammaln((nu+1.)/2.) - gammaln(nu/2.) -(1./2.)*np.log(math.pi) - (1./2.)*np.log(nu) - np.log(sigma) - ((nu+1.)/2.)*np.log(1. + (((indData-mu)/sigma)**2.)/nu))
    return result

def loglikelihoodG(alpha,beta):
    result = sum(-0.5*np.log(2.*math.pi) - np.log(alpha) - (indData-beta)*(indData-beta)/2./alpha/alpha)
    return result

N=1000                     # number of samples
autoL=1                   # autocoreelation length to get independet samples
input = np.genfromtxt("data.dat", delimiter=" ")
indData=input[::100]
datL = len(indData)
length=N*autoL
data=np.zeros((length,8))
accept = 0
Taccept=0
sampalpha = 1.
likehood=1.

# ST eigenvectors of fisher matrix, calculated analytically in mathematica
evecS = np.zeros((3,3),dtype=float)
evecS[0] = [(1./6.)*(65. - 3.*math.pi**2. - math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.)) , 0. , 1.]
evecS[1] = [0. , 1. , 0.]
evecS[2] = [(1./6.)*(65. - 3.*math.pi**2. + math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.)) , 0. , 1.]

# gaussian eigenvectors of fisher matrix, calculated analytically in mathematica
evecG = np.zeros((2,2),dtype=float)
evecG[0] = [-1. , 0.]
evecG[1] = [0. , -1.]

# ST eigenvalues of fisher matrix, calculated analytically from mathemaica
evalueS = [0.,0.,0.]
evalueS[2] = (1./72.)*(7. + 3.*math.pi**2. + math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.))
evalueS[1] = 2./3.
evalueS[0] = (1./72.)*(7. + 3.*math.pi**2. - math.sqrt(4261. - 390.*math.pi**2. + 9.*math.pi**4.))

# gaussian eigenvalues of fisher matrix, calculated analytically from mathemaica
evalueG = [0.,0.]
evalueG[0] = 2.66682
evalueG[1] = 1.33341


# ST initial guess for parameters (use the correct ones we know for know)
paramS = [0,0,0]
newpS = [0.,0.,0.]
paramS[0] = 2.        # nu
paramS[1] = 1.        # mu
paramS[2] = 1.        # sigma

# G initial guess for parameters (use the correct ones we know for know)
paramG = [0,0,0]
newpG = [0.,0.,0.]
paramG[0] = paramS[2]*math.sqrt(paramS[0]/(paramS[0] + 1.))    # alpha, initial guess with alpha = sigma*sqrt(nu/(nu+1.))
paramG[1] = paramS[1]       # beta
paramG[2] = 0.       # null position

# output parameters
outs = [0.,1.,1.]

#
# loop for mcmc sampling of the distribution
#

# start with the ST model and go from there
oldlike = loglikelihoodS(paramS[0],paramS[1],paramS[2])
mnum = 0   # 0 = ST, 1 = Gaussian

for i in range(0,length):

    accept = 0
    # proposing a transdimensional jump
    tranjump = npr.uniform(0,1,1)[0]
    if (tranjump < 0.5):
        changeM = 1
    else:
        changeM = 0

    data[i,1] = changeM

    if (changeM == 1):   # model changed, calculate the transdimensional jumps
        if (mnum == 0): # switching from ST to Gaussian
            j = np.random.choice((0,1))
            newpG[0] = paramS[2]*math.sqrt(paramS[0]/(paramS[0] + 1.)) + npr.normal(0,sampalpha,1)[0]*evecG[j][0]/math.sqrt(evalueG[j])
            newpG[1] = paramS[1] + npr.normal(0,sampalpha,1)[0]*evecG[j][1]/math.sqrt(evalueG[j])
            u = npr.uniform(0,1,1)[0]
            
            if (newpG[0]>.1 and newpG[0]<10. and newpG[1]>-5. and newpG[1]<5.):
            #inside prior range, proceed to calculating MH ratio

                # calculate the Metropolis-Hastings ratio for the new and old sample
                newlike = loglikelihoodG(newpG[0],newpG[1])
                MH_ratio = newlike - oldlike

                # determine whether or not to keep the new sample
                if (np.log(u)<MH_ratio and MH_ratio > -np.inf): # Accept because of MH ratio
                    paramG[0] = newpG[0]
                    paramG[1] = newpG[1]
                    paramG[2] = newpG[2]
                    outs[0]=paramG[0]
                    outs[1]=paramG[1]
                    outs[2]=paramG[2]
                    likehood = newlike
                    oldlike=newlike
                    Taccept=1
                    mnum = 1
                else:
                    likehood = oldlike
                    Taccept=0
        
            else:
                Taccpet = 0
                likehood = oldlike


        else: # switching from Gaussian to ST
            j = np.random.choice((0,1,2))
            newpS[0] = paramS[0] + 0.05*npr.normal(0,sampalpha,1)[0]*(evecS[j][0])/math.sqrt(evalueS[j]*3.)
            newpS[1] = paramG[1] + 0.05*npr.normal(0,sampalpha,1)[0]*(evecS[j][1])/math.sqrt(evalueS[j]*3.)
            newpS[2] = paramG[0]*math.sqrt((newpS[0]+1.)/newpS[0])
#            print newpS
            u = npr.uniform(0,1,1)[0]
            
            if (newpS[0]>.1 and newpS[0]<10. and newpS[1]>-5. and newpS[1]<5. and newpS[2]>0.1 and newpS[2]<10.):

                # calculate the Metropolis-Hastings ratio for the new and old sample
                newlike = loglikelihoodS(newpS[0],newpS[1],newpS[2])
                MH_ratio = newlike - oldlike
                
                # determine whether or not to keep the new sample
                if (np.log(u)<MH_ratio and MH_ratio > -np.inf):
                    paramS[0] = newpS[0]
                    paramS[1] = newpS[1]
                    paramS[2] = newpS[2]
                    outs[0]=paramS[0]
                    outs[1]=paramS[1]
                    outs[2]=paramS[2]
                    likehood = newlike
                    oldlike=newlike
                    Taccept=1
                    mnum=0
                else:
                    likehood = oldlike
                    Taccept=0

            else:
                Taccpet = 0
                likehood = oldlike
        
        
            
    else:   # model not change, continue with normal propsoal inisde the current model
        if (mnum == 0): # ST model
            j = np.random.choice((0,1,2))
            newpS[0] = paramS[0] + 0.05*npr.normal(0,sampalpha,1)[0]*(evecS[j][0])/math.sqrt(evalueS[j]*3.)
            newpS[1] = paramS[1] + 0.05*npr.normal(0,sampalpha,1)[0]*(evecS[j][1])/math.sqrt(evalueS[j]*3.)
            newpS[2] = paramS[2] + 0.05*npr.normal(0,sampalpha,1)[0]*(evecS[j][2])/math.sqrt(evalueS[j]*3.)
            u = npr.uniform(0,1,1)[0]

            if (newpS[0]>.1 and newpS[0]<10. and newpS[1]>-5. and newpS[1]<5. and newpS[2]>0.1 and newpS[2]<10.):

                # calculate the Metropolis-Hastings ratio for the new and old sample
                newlike = loglikelihoodS(newpS[0],newpS[1],newpS[2])
                MH_ratio = newlike - oldlike

                # determine whether or not to keep the new sample
                if (np.log(u)<MH_ratio and MH_ratio > -np.inf):
                    paramS[0] = newpS[0]
                    paramS[1] = newpS[1]
                    paramS[2] = newpS[2]
                    outs[0]=paramS[0]
                    outs[1]=paramS[1]
                    outs[2]=paramS[2]
                    likehood = newlike
                    oldlike=newlike
                    accept=1
                else:
                    likehood = oldlike
                    accept=0

            else:
                accpet = 0
                likehood = oldlike
    
    
        else:   # gaussian model
            j = np.random.choice((0,1))
            newpG[0] = paramG[0] + npr.normal(0,sampalpha,1)[0]*(evecG[j][0])/math.sqrt(evalueG[j])
            newpG[1] = paramG[1] + npr.normal(0,sampalpha,1)[0]*(evecG[j][1])/math.sqrt(evalueG[j])
            u = npr.uniform(0,1,1)[0]
            
            if (newpG[0]>.1 and newpG[0]<10. and newpG[1]>-5. and newpG[1]<5.):
            #inside prior range, proceed to calculating MH ratio

                # calculate the Metropolis-Hastings ratio for the new and old sample
                newlike = loglikelihoodG(newpG[0],newpG[1])
                MH_ratio = newlike - oldlike

                # determine whether or not to keep the new sample
                if (np.log(u)<MH_ratio and MH_ratio > -np.inf): # Accept because of MH ratio
                    paramG[0] = newpG[0]
                    paramG[1] = newpG[1]
                    paramG[2] = newpG[2]
                    outs[0]=paramG[0]
                    outs[1]=paramG[1]
                    outs[2]=paramG[2]
                    likehood = newlike
                    oldlike=newlike
                    accept=1
                else:   # reject because of MH ratio
                    likehood = oldlike
                    accept=0

            else:   # reject because outside prior range
                accpet = 0
                likehood = oldlike

    data[i,0] = mnum
    data[i,1] = changeM
    data[i,2] = Taccept*changeM
    data[i,3] = outs[0]
    data[i,4] = outs[1]
    data[i,5] = outs[2]
    data[i,6] = likehood
    data[i,7] = accept

np.savetxt('test.dat',data,fmt='%i %i %i %f %f %f %f %i',delimiter='\t')

#
# Post processing
#

## getting the acceptance fraction
#acceptance=100*sum(data[:,5])/length
#print acceptance
#
## finding the atucorrelation length
#
#p1 = data[:,1]
#p2 = data[:,2]
#p3 = data[:,3]
#C1 =  np.correlate((p1 - np.mean(p1)),(p1 - np.mean(p1)),mode='full')[(length -1):]
#C2 =  np.correlate((p2 - np.mean(p2)),(p2 - np.mean(p2)),mode='full')[(length -1):]
#C3 =  np.correlate((p3 - np.mean(p3)),(p3 - np.mean(p3)),mode='full')[(length -1):]
#cor1=C1/(np.max(C1))
#cor2=C2/(np.max(C2))
#cor3=C3/(np.max(C3))
#autol = [0,0,0]
#autol[0] = np.where(cor1<0.01)[0]
#autol[1] = np.where(cor2<0.01)[0]
#autol[2] = np.where(cor3<0.01)[0]
#
#print autol[0][0]
#print autol[1][0]
#print autol[2][0]
#
#plt.figure(1)
#plt.plot(p1,p3,'.')
#plt.savefig('nu-sigma.pdf')
#plt.figure(2)
#plt.plot(p1,p2,'.')
#plt.savefig('nu-mu.pdf')
#plt.figure(3)
#plt.plot(p2,p3,'.')
#plt.savefig('mu-sigma.pdf')








