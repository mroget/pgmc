import numpy as np
import scipy as sp
import copy
import sklearn as sk
import torch
from sklearn.base import ClassifierMixin, BaseEstimator
from multiprocess import Pool

import time

from numba import jit
import numba

#from pgmc import embeddings
import src.pgmc.embeddings as embeddings

@jit(nopython=True)
def sqrtm_jit(A):
    U, s, Vh = np.linalg.svd(A)
    s = np.sqrt(s)
    return U @ np.diag(s) @ Vh

@jit(nopython=True)
def _get_rho_jit(x):
    """
    mat = np.zeros((len(x),len(x)),dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(x)):
            mat[i,j] = x[i]*x[j]
    return mat / sum([i**2 for i in x])
    """
    #x = np.ascontiguousarray(x)
    y = np.reshape(x,(len(x),1))
    return (y @ y.T)

@jit(nopython=True)
def _embed_jit(X, embedding):
    ret = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X[i])):
            ret[i,j] = X[i,j]
    for i in range(len(ret)):
        tmp = embedding(ret[i])
        for j in range(len(ret[i])):
            ret[i,j] = tmp[j]
    return ret
"""
@jit(nopython=True)
def _barycentre(X):
    assert(len(X)>0)
    rho = np.zeros((len(X[0]),len(X[0])),dtype=np.float64)
    for x in X:
        rho = rho + _get_rho_jit(x)
    return rho / len(X)
"""
@jit(nopython=True)
def _group_by_jit(X,y):
    sizes = [0 for i in range(len(set(y)))]
    dim = len(X[0])
    for i in y:
        sizes[i]+=1
    data = [np.zeros((sizes[i],dim),dtype=np.float64) for i in range(len(sizes))]
    counters = [0 for i in range(len(sizes))]
    for i in range(len(X)):
        k = y[i]
        data[k][counters[k]] = X[i]/sum([val**2 for val in X[i]])
        counters[k]+=1
    return data

def _barycentre(X):
    assert(len(X)>0)
    return np.einsum("li,lj -> ij",X,X)/len(X)


def _get_rho_k_jit(data, dim):
    rho_k = np.zeros((len(data),dim,dim),dtype=np.float64)
    for i in range(len(data)):
        rho_k[i]  = _barycentre(data[i])
    return rho_k



def sqrtm_inv_numpy(A):
    U, s, Vh = torch.linalg.svd(A)
    s = 1./np.sqrt(s)
    return U @ torch.diag(s) @ Vh

def _pgmc_fit_numpy(data, p):
    p = torch.tensor(p)
    rho_k = []
    for X in data:
        rho_k.append(np.einsum("li,lj -> ij",X,X)/len(X))
    rho_k = torch.tensor(np.array(rho_k))
    rho = torch.einsum("k,kij -> ij", p, rho_k)
    rho = sqrtm_inv_numpy(rho)

    E = torch.einsum("k,il,klm,mj -> kij", p, rho, rho_k, rho)
    return E

def _pgmc_proba_numpy(X, E):
    X_ = torch.tensor(X)
    rho = torch.einsum("li,lj -> lij",X_,X_)
    proba = torch.einsum("kij,nij -> nk", E, rho)
    return proba


def _pgmc_fit(X, y, K, p):
    dim = len(X[0])
    dic = {K[i]:i for i in range(len(K))}
    y_ = np.array([dic[i] for i in y])
    data = _group_by_jit(X,y_)
    return _pgmc_fit_numpy(data, p)



class PGMC(BaseEstimator, ClassifierMixin):
    """ Class For the k-PGMC algorithm
    This class implement the k-PGMC algorithm including classe's weights optimization and kernel/embedding specification.
    The pseudo inverse (necessary for the K-PGMC) use pytorch subroutine. If you wish to run it on cuda it is possible by specifying the device.
    This class is sklearn compliant and can be used with sklearn functions like k-fold crossvalidation or pipeline.
    """
    def __init__(self, embedding=None, class_weight_method=None, class_weight=None, copies=1, device = "cpu"):
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
        self.class_weight = {}
        self.copies = copies
        self.class_weight_method = class_weight_method
        self.device=device

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

        
        # For JIT compilation
        X = np.array([[1.,1.],[3.,2.]])
        y = np.array([0,1])
        self.fit(X,y)
        self.predict(X)


    def _adjust_class_weight(self):
        # Adjust class weights
        if self.class_weight_method == "auto":
            self.class_weight = {k: min(max(2/len(self.K)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.K}
        elif self.class_weight_method == "fast_optimize":
            self._optimize_class_weight_fast()
        elif self.class_weight_method == "optimize":
            self._optimize_class_weight_kfold()
        else:
            self.class_weight = {k:1. for k in self.K}
        

    def _optimize_class_weight_fast(self):
        # Optimization of the weights
        self.class_weight = {k: min(max(2/len(self.K)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.K}
        clf = KPGMC(embedding=self.embedding, class_weight_method=None)
        def f_(x):
            clf.fit(self.X,self.y,class_weight={K[i]:x[i] for i in range(len(K))})
            y_pred.append(clf.predict(self.X))
            return 1.- sk.metrics.f1_score(self.y,y_pred,average="binary" if len(self.K)==2 else "micro")
        default = f_(np.array(list(self.class_weight.values())))
        res = sp.optimize.minimize(f_,np.array(list(self.class_weight.values())),bounds=[(0,1)]*len(self.K),method="Nelder-Mead")
        self.class_weight = {self.K[i]:res.x[i] for i in range(len(self.K))}


    def _optimize_class_weight_kfold(self):
        # Optimization of the weights
        self.class_weight = {k: min(max(2/len(self.K)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.K}
        clf = KPGMC(embedding=self.embedding, class_weight_method=None)
        V = []
        y_true = []
        rs = sk.model_selection.ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        for i, (train_index, test_index) in enumerate(rs.split(self.X)):
            V.append(clf._predict_proba(self.X[test_index]))
            y_true.append(self.y[test_index])
        V = np.concatenate(V)
        y_true = np.concatenate(y_true)

        def f_(x):
            y_true = []
            y_pred = []
            for i, (train_index, test_index) in enumerate(rs.split(self.X)):
                clf.fit(self.X[train_index],self.y[train_index],class_weight={K[i]:x[i] for i in range(len(K))})
                y_pred.append(clf.predict(self.X[test_index]))
                y_true.append(self.y[test_index])
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            return 1.- sk.metrics.f1_score(y_true,y_pred,average="binary" if len(self.K)==2 else "micro")
        default = f_(np.array(list(self.class_weight.values())))
        res = sp.optimize.minimize(f_,np.array(list(self.class_weight.values())),bounds=[(0,1)]*len(self.K),method="Nelder-Mead")
        self.class_weight = {self.K[i]:res.x[i] for i in range(len(self.K))}
        
        

    def fit(self, X, y, class_weight=None):
        """ Fit classifier according to dataset X,y
        Parameters:
            - X (numpy 2d array shape=(n,d)) : training vectors
            - y (numpy 1d array shape=(n,)) : training classes
        Return:
            Nothing
        """
        assert(len(X)==len(y))

        self.K = list(set(y)) # Classes
        self.X = _embed_jit(X, self.embedding)
        self.y = copy.deepcopy(y)

        # Group by class
        dim = len(self.X[0])
        dic = {self.K[i]:i for i in range(len(self.K))}
        y_ = np.array([dic[i] for i in self.y])
        data = _group_by_jit(self.X,y_)

        # class weights
        if class_weight == None:
            self._adjust_class_weight()
        else:
            self.class_weight = class_weight

        p = np.array([self.class_weight[self.K[i]] for i in range(len(self.K))],dtype=np.float64)
        
        # Fit
        p = torch.tensor(p, device=self.device)
        rho_k = []
        for X in data:
            rho_k.append(np.einsum("li,lj -> ij",X,X)/len(X))
        rho_k = torch.tensor(np.array(rho_k), device=self.device)
        rho = torch.einsum("k,kij -> ij", p, rho_k)
        rho = sqrtm_inv_numpy(rho)

        self.E = torch.einsum("k,il,klm,mj -> kij", p, rho, rho_k, rho)

    def _predict_proba(self, X):
        X_ = torch.tensor(_embed_jit(X, self.embedding), device=self.device)
        rho = torch.einsum("li,lj -> lij",X_,X_)
        proba = torch.einsum("kij,nij -> nk", self.E, rho)
        return proba
        
    def predict(self, X):
        """ Predict classes of sample X
        Parameter:
            - X (numpy 2d array shape=(n,d)) : sample to classify
        Return:
            - y (numpy 1d array shape=(n,)) : classes predicted by the model

        """
        t = time.time()
        p = self._predict_proba(X)
        t = time.time() - t
        print("predict", t)
        y_pred = np.array([self.K[np.argmax(x)] for x in p])
        return y_pred
        
    def predict_proba(self, X):
        """ Predict class probablities of sample X
        Parameter:
            - X (numpy 2d array shape=(n,d)) : sample to classify
        Return:
            - Py (numpy 2d array sape=(n,d)) : Probability for each point of X to be in each class according to the model.

        """
        p = self._predict_proba(X)
        
        return [{self.K[j]:p[i,j] for j in range(len(p[i]))} for i in range(len(p))]