import numpy as np
import scipy as sp
import copy
import sklearn as sk
import torch
from sklearn.base import ClassifierMixin, BaseEstimator

from numba import jit

#from pgmc import embeddings
import src.pgmc.embeddings as embeddings

@jit
def dot(x,y):
    return np.dot(x,y)


def kernel_mat(kernel, X, kernel_parameter):
    Kernel_Matrix = np.zeros((len(X),len(X)),dtype=float)
    for i in range(len(X)):
        for j in range(i,len(X)):
            if kernel_parameter == None:
                Kernel_Matrix[i,j] = kernel(X[i],X[j])
            else:
                Kernel_Matrix[i,j] = kernel(X[i],X[j],kernel_parameter)
            Kernel_Matrix[j,i] = Kernel_Matrix[i,j]
    return Kernel_Matrix



class KPGMC(BaseEstimator, ClassifierMixin):
    """ Class For the k-PGMC algorithm
    This class implement the k-PGMC algorithm including classe's weights optimization and kernel/embedding specification.
    The pseudo inverse (necessary for the K-PGMC) use pytorch subroutine. If you wish to run it on cuda it is possible by specifying the device.
    This class is sklearn compliant and can be used with sklearn functions like k-fold crossvalidation or pipeline.
    """
    def __init__(self, kernel=None, embedding=None, class_weight=None, kernel_parameter=None, device="cpu"):
        """ Initialisation of a new KPGMC
        Parameters:
            - kernel (function, default=None) : The kernel used to compute similarity between vectors (default is scalar product). A kernel with high values might cause the svd to diverge.
            - embedding (str or function, default=None) : The embedding method (default is normalization). 
                + "normal" : Normalization of the vectors.
                + "stereo" : Inverse stereoscopic embedding.
                + "orthogonal" : Vectors are normalized, then a new features of value -1 is added, then the vectors are normalized again.
            - class_weight (dict or str, default None) : The method to choose the classes's weights. By default, all classes have weight 1. A dictionnary {class : weight} is accepted.
                + "auto" : Raisonnable weights are chosen according to some formula. Fast but not optimal.
                + "optimize" : The weight are optimized on the training dataset (for faster computation) using Nelder-Mead method.
            - device (str, default="cpu") : the device on which pytorch run the pinv. For gpu computation, specify "cuda" instead.  
        Return:
            A KPGMC classifier.

        """
        self.class_weight = class_weight
        self.device = device
        self.kernel_parameter = kernel_parameter

        if kernel==None:
            self.kernel = dot
        else:
            self.kernel = kernel

        if embedding==None:
            self.embedding = embeddings.normalize
        elif type(embedding) == str and embedding.lower() == "normal":
            self.embedding = embeddings.normalize
        elif type(embedding) == str and embedding.lower() == "stereo":
            self.embedding = embeddings.stereo
        elif type(embedding) == str and embedding.lower() == "orthogonal":
            self.embedding = embeddings.orthogonalize
        else:
            self.embedding = embedding

        kernel_mat(self.kernel, np.array([[1.]]), self.kernel_parameter) # For JIT compilation


    def _adjust_class_weight(self):
        # Adjust class weights
        if type(self.class_weight) == dict:
            pass
        elif self.class_weight == "auto":
            self.class_weight = {k: min(max(2/len(self.K)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.K}
        elif self.class_weight == "optimize":
            self._optimize_class_weight()
        else:
            self.class_weight = {k:1. for k in self.K}

        self.T = torch.tensor(np.array([self.class_weight[self.K[i]] * self.POVM[i] for i in range(len(self.K))]),device=self.device)

    def _compute_matrices_pi(self):
        # Compute the matrix used for classification
        """
        self.Kernel_Matrix = np.zeros((len(self.X),len(self.X)),dtype=float)
        for i in range(len(self.X)):
            for j in range(i,len(self.X)):
                self.Kernel_Matrix[i,j] = self.kernel(self.X[i],self.X[j])
                self.Kernel_Matrix[j,i] = self.Kernel_Matrix[i,j]
        """
        self.Kernel_Matrix = kernel_mat(self.kernel, self.X, self.kernel_parameter)
        Ginv = sp.linalg.sqrtm(torch.linalg.pinv(torch.tensor(self.Kernel_Matrix,device=self.device)).cpu())
        
        self.POVM = []                                       
        for k in self.K:
            self.POVM.append(np.real(np.dot(Ginv,np.dot(np.diag([1. if i==k else 0. for i in self.y]),Ginv))))

    def _optimize_class_weight(self):
        # Optimization of the weights
        self.class_weight = {k: min(max(2/len(self.K)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.K}
        V = np.array([[np.dot(np.dot(self.Kernel_Matrix[i].transpose(),self.POVM[k]),self.Kernel_Matrix[i]) for k in range(len(self.K))] for i in range(len(self.X))],dtype=float)
        def f_(x):
            P = copy.deepcopy(V)
            for i in range(len(x)):
                P[:,i] *= x[i]
            y_pred = np.array([np.argmax(p) for p in P],dtype=float)
            return 1.- sk.metrics.f1_score(self.y,y_pred,average="binary" if len(self.K)==2 else "micro")
        default = f_(np.array(list(self.class_weight.values())))
        res = sp.optimize.minimize(f_,np.array(list(self.class_weight.values())),bounds=[(0,1)]*len(self.K),method="Nelder-Mead")
        self.class_weight = {self.K[i]:res.x[i] for i in range(len(self.K))}
        
        

    def fit(self, X, y):
        """ Fit classifier according to dataset X,y
        Parameters:
            - X (numpy 2d array shape=(n,d)) : training vectors
            - y (numpy 1d array shape=(n,)) : training classes
        Return:
            Nothing
        """
        assert(len(X)==len(y))
        self.K = list(set(y)) # Classes
        self.X = np.array(list(map(self.embedding,X)))
        self.y = copy.deepcopy(y)

        self._compute_matrices_pi()
        self._adjust_class_weight()
        
    def _predict_proba_one(self,w):
        if self.kernel_parameter == None:
            W = torch.tensor(np.array([self.kernel(x,w) for x in self.X],dtype=float),device=self.device)
        else:
            W = torch.tensor(np.array([self.kernel(x,w,self.kernel_parameter) for x in self.X],dtype=float),device=self.device)
        p = torch.tensordot(W,torch.tensordot(self.T,W,dims=([2], [0])),dims=([0], [1]))
        return np.array(p.cpu(),dtype=float)

    def _predict_one(self, w):
        i = np.argmax(self._predict_proba_one(w))
        return self.K[i]

    def predict(self, X):
        """ Predict classes of sample X
        Parameter:
            - X (numpy 2d array shape=(n,d)) : sample to classify
        Return:
            - y (numpy 1d array shape=(n,)) : classes predicted by the model

        """
        return np.array(list(map(self._predict_one,np.array(list(map(self.embedding,X))))),dtype=float)
        
    def predict_proba(self, X):
        """ Predict class probablities of sample X
        Parameter:
            - X (numpy 2d array shape=(n,d)) : sample to classify
        Return:
            - Py (numpy 2d array sape=(n,d)) : Probability for each point of X to be in each class according to the model.

        """
        return np.array(list(map(self._predict_proba_one,np.array(list(map(self.embedding,X))))),dtype=float)
