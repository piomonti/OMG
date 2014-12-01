### Python implementation of Online Mixture of (penalised) Gaussians (OMG)
#
#
#

import math
import numpy
import os
import pandas
import multiprocessing
from sklearn.preprocessing import scale
from sklearn.covariance import graph_lasso
from OfflineMoG import PenOfflineEM, DlogLike

class OMG():
    """Class for Online Mixture of (penalised) Gaussians (OMG) algorithm
    
    """
    
    def __init__(self, Files, K, rho, NburnIn=10, BurnInIters=5,  tol=.01, max_iter=50, verbose=False):
	"""
	
	INPUT:
         - Files:  list of file directory for BOLD time series of each subject or a list of data matrices for each subject
                   If Files is a list of file directories then these will only be loaded on a need to basis resulting in efficient use of CPU memory
         - K: number of clusters
         - rho: regularisation parameter
         - NburnIn: number of subjects to use during burn in
         - BurnInIters: number of times to run initial penalised MoG algorithm (ie initial clustering)
         - tol: convergence criteria at each step
         - max_iter: maximum number of iterations at each step
         - verbose: binary flag to indicate if progress should be printed
	
	
	"""
	
	self.Files = Files
	self.K = K
	self.rho = rho
	self.NburnIn = NburnIn
	self.BurnInIters = BurnInIters
	self.tol = tol
	self.max_iter = max_iter
	self.verbose = verbose
	self.SubjectCount = 0 # keep track of subjects that have already been used
	self.C = None
	
	if type(self.Files[0])==str:
	    self.FileType = 'str'
	    # load in reduced subset of data for burnin
	    self.BurnInData = [scale(loadData(Files[i])) for i in range(self.NburnIn)]
	    if self.verbose: print "File directories provided - will only be loaded when needed"
	else:
	    self.FileType = 'data'
	    self.BurnInData = [scale(self.Files[i]) for i in range(self.NburnIn)]
	    if self.verbose: print "Subject time series provided apriori"
	    
    
    def __repr__(self):
	""""""
	mes = '### Online Mixture of (penalised) Gaussians ###\n'
	mes += '# K: '+ str(self.K) +'\n'
	mes += '# reg: '+ str(self.rho) +'\n'
	return mes
    
    def fitBurnIn(self):
	"""
	Run offline penalised MoG algorithm and obtain initial estaimtes
	
	Due to non-convex nature of MoG, we run this initial step several times and choose value minimising BIC
	
	"""
	
	if self.verbose: print "Running initial burn in. Will run " +str(self.BurnInIters) + " burn in iterations"
	
	# run first iteration to set benchmark:
	failFlag = False
	try:
	    [self.Theta, self.C, self.Z, self.w, self.BIC, iter_] = PenOfflineEM(D=self.BurnInData, K=self.K, rho=self.rho, tol=self.tol, max_iter=self.max_iter, verbose=False) # run offline EM on burn in subjects
	except ValueError:
	    failFlag = True
	    if self.verbose: print "Only one cluster found... consider increasing burnin size"
	    self.BIC = numpy.Inf # this will be replaced by any fitted model!
	
	for i in range(1,self.BurnInIters):
	    if self.verbose: print "Running burn in iteration: " + str(i)
	    failFlag = False
	    try:
		[Theta, C, Z, w, BIC, iter_] = PenOfflineEM(D=self.BurnInData, K=self.K, rho=self.rho, tol=self.tol, max_iter=self.max_iter, verbose=False) # run offline EM on burn in subjects
	    except ValueError:
		failFlag = True
		if self.verbose: print "Only one cluster found... consider increasing burnin size"
	    if (failFlag==False):
		if BIC < self.BIC:
		    # lower BIC value - save these results:
		    self.Theta = numpy.copy(Theta)
		    self.C = numpy.copy(C)
		    self.Z = numpy.copy(Z)
		    self.w = numpy.copy(w)
		    self.BIC = numpy.copy(BIC)
	# define additional parameters (mainly store sample covariance for each cluster as this will be needed for updating precisions!):
	self.S = numpy.zeros((self.Theta.shape[0], self.Theta.shape[1], self.Theta.shape[1])) # store sample covariance matrix for each cluster!
	for i in range(len(self.BurnInData)):
	    for j in range(self.Theta.shape[0]):
		self.S[j,:,:] += numpy.cov(self.BurnInData[i], rowvar=0, bias=1) * self.Z[i,j]/self.w[j]
	self.SubjectCount = self.NburnIn

    
    def UpdateSubject(self, subjectCount = None):
	"""
	Update parameters based on given subject
	
	Input:
	     - subjectCount: which element from self.Files should be added next. If is None, then self.SubjectCount is used
	
	"""
	
	if subjectCount==None:
	    subjectCount = self.SubjectCount
	
	if self.verbose: print "Updating with Subject: "+str(subjectCount)
	
	if self.FileType=='str':
	    # load in subjects data:
	    newData = scale(loadData(self.Files[subjectCount]))  
	else:
	    # data already loaded
	    newData = scale(self.Files[subjectCount])
	    
	# begin update:
	convergence = False
	iter_ = 0
	#Theta = numpy.copy(self.Theta)
	ThetaOld = numpy.copy(self.Theta)
	p = self.Theta.shape[1] # number of nodes
	K = self.Theta.shape[0] # number of clusters
	
	# run EM algorithm for new subject!
	while (convergence==False) & (iter_ < self.max_iter):
	    ## -- E step (i.e., choose which cluster this subject belongs to):
	    wNew = numpy.array([DlogLike(newData, Pres=self.Theta[x,:,:]) for x in range(K)])
	    wNew -= max(wNew) # center around maximum value (to avoid floating point errors)
	    wNew = numpy.exp(wNew) * self.w # exponentiate and multiply by mixing probabilities
	    # finally normalise
	    wNew /= sum(wNew)
	
	    wUpdate = wNew + self.w # updated mixing distribution (unnormalised!)
	    
	    ## -- M step (i.e., update precision estimates for each cluster)
	    Snew = numpy.zeros((K,p,p))
	    for k in range(K):
		Snew[k,:,:]  = (self.S[k,:,:] * self.w[k] + wNew[k] * numpy.cov(newData, rowvar=0, bias=1) )/wUpdate[k]
		self.Theta[k,:,:] = graph_lasso(emp_cov = Snew[k,:,:], alpha=self.rho)[1]
	    
	    # check convergence:
	    if abs(self.Theta-ThetaOld).sum() < self.tol:
		convergence=True
		# update all sample statistics that we're storing, starting with covariance:
		for i in range(K):
		    self.S[k,:,:] = numpy.copy(Snew[k,:,:])
		self.w = numpy.copy(wUpdate)
		self.Z = numpy.vstack((self.Z, wNew))
		self.SubjectCount += 1 # keep track of which subject we are on
		self.C = numpy.hstack((self.C, wNew.argmax()))
	    else:
		iter_ += 1
		ThetaOld = numpy.copy(self.Theta)
		

def loadData(FileDir):
    """Read in data from a given directory"""
    #with open (FileDir, 'rb') as csvfile:
	#d = numpy.genfromtxt(csvfile, dtype=None)
    return numpy.array(pandas.read_csv(FileDir))
