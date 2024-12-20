import numpy as np
import scipy as sp
import copy
import sklearn as sk
import torch
from sklearn.base import ClassifierMixin, BaseEstimator

import time

from numba import jit
import numba

from pgmc import embeddings
#import src.pgmc.embeddings as embeddings

@jit(nopython=True)
def _embed_jit(X, embedding):
    d = len(embedding(X[0]))
    ret = np.zeros((len(X), d))
    for i in range(len(ret)):
        tmp = embedding(X[i])
        for j in range(len(tmp)):
            ret[i,j] = tmp[j]
    return ret

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
        data[k][counters[k]] = X[i]
        counters[k]+=1
    return data

def kron_power(A, n):
    l = list(map(int,list(bin(n)[2:])))[::-1]
    mat = [A]
    while len(mat) < len(l):
        mat.append(np.kron(mat[-1],mat[-1]))

    ret = np.array([[1.]])
    for i in range(len(l)):
        if l[i]==1:
            ret = np.kron(ret,mat[i])
    return ret

def sqrtm_inv_torch(A):
    U, s, Vh = torch.linalg.svd(A)
    s = 1./np.sqrt(s)
    return U @ torch.diag(s) @ Vh


class PGMC(BaseEstimator, ClassifierMixin):
    """ Class For the k-PGMC algorithm
    This class implement the k-PGMC algorithm including classe's weights optimization and kernel/embedding specification.
    The pseudo inverse (necessary for the K-PGMC) use pytorch subroutine. If you wish to run it on cuda it is possible by specifying the device.
    This class is sklearn compliant and can be used with sklearn functions like k-fold crossvalidation or pipeline.
    """
    def __init__(self, embedding=None, class_weight_method=None, copies=1, device = "cpu"):
        """ Initialisation of a new KPGMC
        Parameters:
            - embedding (str or function, default=None) : The embedding method (default is normalization). 
                + "normal" : Normalization of the vectors.
                + "stereo" : Inverse stereoscopic embedding.
                + "orthogonal" : Vectors are normalized, then a new features of value -1 is added, then the vectors are normalized again.
            - class_weight_method (dict or str, default None) : The method to choose the classes's weights. By default, all classes have weight 1. A dictionnary {class : weight} is accepted.
                + "proportional" : Set a weight proportional to the presence of each class in the training dataset.
                + "auto" : Raisonnable weights are chosen according to some formula for binary dataset. Proportional weights are chosen for more than two classes.
                + "optimize" : The weight are optimized on the training dataset (for faster computation) using Nelder-Mead method.
            - device (str, default="cpu") : the device on which pytorch run the pinv. For gpu computation, specify "cuda" instead. 
            - copies (int, default=1) : the number of copies (increases the running time and memory usage greatly).
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

        self.fitted = False
        self.classes_ = None


    def _adjust_class_weight(self):
        # Adjust class weights
        if self.class_weight_method == "auto":
            if len(self.classes_) == 2:
                self.class_weight = {k: min(max(2/len(self.classes_)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.classes_}
            else:
                self.class_weight = {k: 1/list(self.y).count(k) for k in self.classes_}
        elif self.class_weight_method == "proportional":
            self.class_weight = {k: 1/list(self.y).count(k) for k in self.classes_}
        elif self.class_weight_method == "optimize":
            self._optimize_class_weight_kfold()
        else:
            self.class_weight = {k:1./len(self.classes_) for k in self.classes_}
        
    def _optimize_class_weight_kfold(self):
        # Optimization of the weights
        self.class_weight = {k: min(max(2/len(self.classes_)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.classes_}
        clf = PGMC(embedding=self.embedding, class_weight_method=None, device=self.device)
        V = []
        y_true = []
        rs = sk.model_selection.ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        for i, (train_index, test_index) in enumerate(rs.split(self.X)):
            V.append(self.X[test_index])
            y_true.append(self.y[test_index])
        V = np.concatenate(V)
        y_true = np.concatenate(y_true)

        def f_(x):
            y_true = []
            y_pred = []
            for i, (train_index, test_index) in enumerate(rs.split(self.X)):
                clf.fit(self.X[train_index],self.y[train_index],class_weights={self.classes_[i]:x[i] for i in range(len(self.classes_))})
                y_pred.append(clf.predict(self.X[test_index]))
                y_true.append(self.y[test_index])
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            return 1.- sk.metrics.f1_score(y_true,y_pred,average="binary" if len(self.classes_)==2 else "micro")
        default = f_(np.array(list(self.class_weight.values())))
        res = sp.optimize.minimize(f_,np.array(list(self.class_weight.values())),bounds=[(0,1)]*len(self.classes_),method="L-BFGS-B")
        self.class_weight = {self.classes_[i]:res.x[i] for i in range(len(self.classes_))}
        
        

    def fit(self, X, y, class_weights=None):
        """ Fit classifier according to dataset X,y
        Parameters:
            - X (numpy 2d array shape=(n,d)) : training vectors
            - y (numpy 1d array shape=(n,)) : training classes
            - class_weights (dict) : For each class, the weight. If left empty, the method specified at the creation of the classifier will be used instead.
        Return:
            Nothing
        Complexity : $O\left(d^c(N + d^{2c}K))\right)$,
        whith 
            - N : number of points
            - K : number of classes
            - d : number of features
            - c : number of copies.
        """
        assert(len(X)==len(y))

        self.classes_ = list(set(y)) # Classes
        self.X = _embed_jit(X, self.embedding)
        self.y = copy.deepcopy(y)

        # Group by class
        dim = len(self.X[0])
        dic = {self.classes_[i]:i for i in range(len(self.classes_))}
        y_ = np.array([dic[i] for i in self.y])
        data = _group_by_jit(self.X,y_)

        # class weights
        if class_weights == None:
            self._adjust_class_weight()
        else:
            self.class_weight = class_weights

        p = np.array([self.class_weight[self.classes_[i]] for i in range(len(self.classes_))],dtype=np.float64)
        
        # Fit
        p = torch.tensor(p, device=self.device)
        rho_k = []
        for X in data:
            rho_k.append(kron_power(np.einsum("li,lj -> ij",X,X), self.copies))
        rho_k = torch.tensor(np.array(rho_k), device=self.device)
        rho = torch.einsum("k,kij -> ij", p, rho_k)
        rho = sqrtm_inv_torch(rho)
        self.E = torch.einsum("k,il,klm,mj -> kij", p, rho, rho_k, rho)

        self.fitted = True

    def _predict_proba(self, X):
        if self.fitted == False or self.classes_==None:
            raise sk.exceptions.NotFittedError


        X_ = torch.tensor(_embed_jit(X, self.embedding), device=self.device)
        rho = torch.einsum("li,lj -> lij",X_,X_)
        proba = torch.einsum("kij,nji -> nk", self.E, rho)
        return proba
        
    def predict(self, X):
        """ Predict classes of sample X
        Parameter:
            - X (numpy 2d array shape=(n,d)) : sample to classify
        Return:
            - y (numpy 1d array shape=(n,)) : classes predicted by the model

        """
        p = self._predict_proba(X)
        y_pred = np.array([self.classes_[np.argmax(x)] for x in p])
        return y_pred
        
    def predict_proba(self, X):
        """ Predict class probablities of sample X
        Parameter:
            - X (numpy 2d array shape=(n,d)) : sample to classify
        Return:
            - Py (numpy 2d array shape=(n,k)) : Probability for each point of X to be in each class according to the model. For one point, the classes are sorted using `self.classes_`.

        """
        p = self._predict_proba(X)
        
        return p