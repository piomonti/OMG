## Offline Mixture of Sparse Multivariate Gaussians 
#
#

import numpy, os, math
from sklearn.covariance import graph_lasso
from sklearn.preprocessing import scale

def PenOfflineEM(D, K, rho, Zinit=None,tol=.1, max_iter=50, verbose=False):
    """Function to fit penalised mixture of Gaussians
    
       INPUT:
	   - D: list of datasets. Each D[[i]] should be an n by p matrix of observations
	   - K: number of clusters
	   - lambda: regulariation penalty 
	   - Zinit: vector of initial cluster estimates
	   - tol: convergence criteria
	   - max_iter: maximum number of iterations 
	
    
    """
    
    # center data (in case this has not already been done):
    D = [scale(x) for x in D]
    
    # define some initial parameters:
    p = D[0].shape[1]
    w = numpy.array([1./K] * K)
    Z = numpy.zeros(( len(D), K))
    convergence = False
    iter_ = 0
    
    # model parameters:
    Theta = numpy.zeros((K, p,p))
    ThetaOld = numpy.zeros((K, p,p)) # used to check convergence
    mu = numpy.zeros((K, p)) # actually this is totally unnecesary if each dataset is centered! same with Dmeans below!
    Dmeans = numpy.array([ x.mean(axis=0) for x in D ]) # mean of each dataset - will be used later
    
    if Zinit==None:
	if verbose: print "Randomly allocating initial clustering"
	Zinit = numpy.random.random(( len(D), K))
	for i in range(len(D)): Z[i, Zinit.argmax(axis=1)[i]]=1
	
    # just in case we normalise Z:
    Z = numpy.apply_along_axis(lambda x: x/sum(x), axis=1, arr=Z)
    if verbose: print numpy.round(Z, decimals=1)
    
    w = Z.mean(axis=0)
    Theta = getPrecision(D=D, Z=Z, rho=rho)
    
    # run EM ALGORITHM!
    while (convergence==False) & (iter_<max_iter):
	
	# **EXPECTATION STEP**:
	for i in range(K):
	    Z[:,i] = [DlogLike(x, Pres=Theta[i,:,:]) for x in D]
	# normalise (remember we are still on the Log scale)
	Z = numpy.apply_along_axis(lambda x: x-x.max(), axis=1, arr=Z) # center results
	Z = numpy.exp(Z) # put back on the likelihood (not loglikelihood) scale
	
	for i in range(K):
	    Z[:,i] *= w[i]
    
	# finally normalise:
	Z = numpy.apply_along_axis(lambda x: x/sum(x), axis=1, arr=Z)
    
	# **MAXIMIZATION STEP**:
	w = Z.mean(axis=0) # update mixing probabilities
	Theta = getPrecision(D=D, Z=Z, rho=rho)
	
	# check convergence:
	if abs(Theta-ThetaOld).sum() < tol:
	    convergence = True
	else:
	    iter_ += 1
	    ThetaOld = numpy.copy(Theta)
	    if verbose: 
		print "Current clustering estimate at iteration:" + str(iter_) + ":"
		print numpy.round(Z, decimals=1)
	    
    C = numpy.apply_along_axis(lambda x: x.argmax(), axis=1, arr=Z) # final clustering
    
    # calculate BIC:
    BIC = getBIC(data=D, Theta=Theta, Z=Z, w=w)
    
    # return *unnormalised mixing distribution* w, this is important for online algorithm!
    w = Z.sum(axis=0)
    
    return [Theta, C, Z, w, BIC, iter_]


def getPrecision(D, Z, rho):
    """M step in EM algorithm - get sparse precision matrices!
    
    INPUT:
         - D: list of datasets
         - Z: latent allocation matrix
         - rho: regulariation parameter
    
    """
    
    K = Z.shape[1] # number of clusters
    p = D[0].shape[1] # number of ROIs
    
    newTheta = numpy.zeros((K, p, p))
    
    for i in range(K):
	# get covariance for ith cluster:
	S = numpy.zeros((p,p))
	for j in range(len(D)):
	    S += numpy.cov(D[j], rowvar=0, bias=1) * Z[j,i]
	S /= Z[:,i].sum()
	
	newTheta[i,:,:] = graph_lasso(emp_cov = S, alpha=rho)[1]

    return newTheta

def DlogLike(data, Pres):
    """
    Get loglikelihood for data array - we do this by summing loglikelihood of each observations (i.e., assuming IID)
    
    """
    return numpy.apply_along_axis(lambda x: logLike(datum=x, Pres=Pres), axis=1, arr=data ).sum()


def logLike(datum, Pres):
    """
    Loglikelihood for one observation (i.e., one vector)
    """
    
    return 0.5*numpy.log(numpy.linalg.det(Pres)) -0.5*numpy.dot(numpy.dot(datum.transpose(), Pres), datum) - 0.5 *Pres.shape[1]*math.log(2*math.pi)


ParamNo = lambda x: (x[numpy.triu_indices(x.shape[0], k=1)] !=0).sum()

def getBIC(data, Theta, Z, w):
    """
    calculate BIC of fitting MoG model
    
    INPUT:
	 - data: list of datasets 
	 - Theta: array of estimated precision matrices
	 - Z: estimate of latent variables
	 - w: mixing distributions
    """
    p = Theta.shape[1] # number of nodes
    K = Theta.shape[0] # number of subjects
    n = data[0].shape[0] # number of observations per subject (assumed constant, used in BIC calculation)
    LL = 0.0 # log likelihood
    
    # calculate loglikelihood of fitted model:
    for i in range(len(data)):
	LLi = sum( numpy.array([ DlogLike(data[i], Pres=Theta[x,:,:]) for x in range(K)]) * Z[i,:]) + sum( numpy.log(w) * Z[i,:])
	LL += LLi
	
    NLL = -2*LL # twice negative log likelihood
    
    # calculate complexity pen:
    ComplexityPen = math.log(n) *  sum([ParamNo(Theta[x,:,:]) for x in range(K)] )
    #numpy.apply_along_axis(lambda x: x.sum(axis=None), axis=2, arr=Theta )
    
    return NLL + ComplexityPen

    
    
    