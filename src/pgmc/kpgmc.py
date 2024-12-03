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



def sqrtm_torch(A):
    U, s, Vh = torch.linalg.svd(A)
    s = torch.sqrt(s)
    return U @ torch.diag(s) @ Vh

@jit
def dot(x,y):
    return np.dot(x,y)

@jit
def kernel_mat(kernel, X):
    Kernel_Matrix = np.zeros((len(X),len(X)),dtype=np.float64)
    for i in range(len(X)):
        for j in range(i,len(X)):
            Kernel_Matrix[i,j] = kernel(X[i],X[j])
            Kernel_Matrix[j,i] = Kernel_Matrix[i,j]
    return Kernel_Matrix

@jit
def kernel_pred(kernel, embedding, X_train, X_pred, device):
    ret = np.zeros((len(X_pred),len(X_train)),dtype=np.float64)
    for j in range(len(X_pred)):
        for i in range(len(X_train)):
            ret[j][i] = kernel(X_train[i],embedding(X_pred[j]))
    return ret


def predict_proba_torch(W, T, device):
    Wt = torch.tensor(W,device=device)
    p1 = torch.tensordot(T,Wt,dims=([2], [1]))
    p = torch.tensordot(p1,Wt,dims=([1], [1]))
    p = torch.diagonal(p, offset=0, dim1=1, dim2=2)
    return np.array(torch.transpose(p,0,1).cpu())



class KPGMC(BaseEstimator, ClassifierMixin):
    """ Class For the k-PGMC algorithm
    This class implement the k-PGMC algorithm including classe's weights optimization and kernel/embedding specification.
    The pseudo inverse (necessary for the K-PGMC) use pytorch subroutine. If you wish to run it on cuda it is possible by specifying the device.
    This class is sklearn compliant and can be used with sklearn functions like k-fold crossvalidation or pipeline.
    """
    def __init__(self, kernel=None, embedding=None, class_weight_method=None, device="cpu"):
        """ Initialisation of a new KPGMC
        Parameters:
            - kernel (function, default=None) : The kernel used to compute similarity between vectors (default is scalar product). A kernel with high values might cause the svd to diverge.
            - embedding (str or function, default=None) : The embedding method (default is normalization). 
                + "normal" : Normalization of the vectors.
                + "stereo" : Inverse stereoscopic embedding.
                + "orthogonal" : Vectors are normalized, then a new features of value -1 is added, then the vectors are normalized again.
            - class_weight_method (dict or str, default None) : The method to choose the classes's weights. By default, all classes have weight 1. A dictionnary {class : weight} is accepted.
                + "proportional" : Set a weight proportional to the presence of each class in the training dataset.
                + "auto" : Raisonnable weights are chosen according to some formula for binary dataset. Proportional weights are chosen for more than two classes.
                + "optimize" : The weight are optimized on the training dataset (for faster computation) using Nelder-Mead method.
            - device (str, default="cpu") : the device on which pytorch run the pinv. For gpu computation, specify "cuda" instead.  
        Return:
            A KPGMC classifier.

        """
        self.class_weight = {}
        self.class_weight_method = class_weight_method
        self.device = device


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

        # For JIT compilation
        X = np.array([[1.],[3.]])
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
        elif self.class_weight_method == "fast_optimize":
            self._optimize_class_weight_fast()
        elif self.class_weight_method == "optimize":
            self._optimize_class_weight_kfold()
        else:
            self.class_weight = {k:1. for k in self.classes_}
        

    def _compute_matrices_pi(self):
        # Compute the matrix used for classification
        self.Kernel_Matrix = torch.tensor(kernel_mat(self.kernel, self.X),device=self.device)
        Ginv = sqrtm_torch(torch.linalg.pinv(self.Kernel_Matrix))
        sigma = torch.tensor(np.array([np.diag([1. if i==k else 0. for i in self.y]) for k in self.classes_]), device = self.device)
        
        self.POVM = torch.transpose(torch.tensordot(Ginv, torch.tensordot(sigma, Ginv, dims=([2], [0])), dims=([1], [1])),0,1)

    def _optimize_class_weight_fast(self):
        # Optimization of the weights
        self.class_weight = {k: min(max(2/len(self.classes_)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.classes_}
        V = np.array(torch.transpose(torch.diagonal(torch.tensordot(torch.tensordot(self.POVM,self.Kernel_Matrix,dims=([2], [1])),self.Kernel_Matrix,dims=([1], [1])), offset=0, dim1=1, dim2=2),0,1).cpu())
        def f_(x):
            P = x * V
            y_pred = np.array([self.classes_[np.argmax(p)] for p in P],dtype=float)
            return 1.- sk.metrics.f1_score(self.y,y_pred,average="binary" if len(self.classes_)==2 else "micro")
        default = f_(np.array(list(self.class_weight.values())))
        res = sp.optimize.minimize(f_,np.array(list(self.class_weight.values())),bounds=[(0,1)]*len(self.classes_),method="Nelder-Mead")
        self.class_weight = {self.classes_[i]:res.x[i] for i in range(len(self.classes_))}


    def _optimize_class_weight_kfold(self):
        # Optimization of the weights
        self.class_weight = {k: min(max(2/len(self.classes_)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.classes_}
        clf = KPGMC(kernel=self.kernel, embedding=self.embedding, class_weight_method=None, device=self.device)
        V = []
        y_true = []
        rs = sk.model_selection.ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        for i, (train_index, test_index) in enumerate(rs.split(self.X)):
            clf.fit(self.X[train_index],self.y[train_index])
            V.append(clf._predict_proba(self.X[test_index]))
            y_true.append(self.y[test_index])
        V = np.concatenate(V)
        y_true = np.concatenate(y_true)

        def f_(x):
            P = x * V
            y_pred = np.array([self.classes_[np.argmax(p)] for p in P],dtype=float)
            return 1.- sk.metrics.f1_score(y_true,y_pred,average="binary" if len(self.classes_)==2 else "micro")
        default = f_(np.array(list(self.class_weight.values())))
        res = sp.optimize.minimize(f_,np.array(list(self.class_weight.values())),bounds=[(0,1)]*len(self.classes_),method="Nelder-Mead")
        self.class_weight = {self.classes_[i]:res.x[i] for i in range(len(self.classes_))}
        
        

    def fit(self, X, y):
        """ Fit classifier according to dataset X,y
        Parameters:
            - X (numpy 2d array shape=(n,d)) : training vectors
            - y (numpy 1d array shape=(n,)) : training classes
        Return:
            Nothing
        """
        assert(len(X)==len(y))
        self.classes_ = list(set(y)) # Classes
        self.X = np.array(list(map(self.embedding,X)))
        self.y = copy.deepcopy(y)

        self._compute_matrices_pi()
        self._adjust_class_weight()
        self.fitted = True


    def _predict_proba(self, X):
        if self.fitted == False or self.classes_==None:
            raise sk.exceptions.NotFittedError

        weights = torch.tensor(list(self.class_weight.values()),device=self.device)
        T = torch.einsum("i,klm->klm",weights,self.POVM)

        W = torch.tensor(kernel_pred(self.kernel, self.embedding, self.X, X, self.device),device=self.device)

        p = np.array(torch.transpose(torch.diagonal(torch.tensordot(torch.tensordot(self.POVM,W,dims=([2], [1])),W,dims=([1], [1])), offset=0, dim1=1, dim2=2),0,1).cpu())

        p = np.array(list(self.class_weight.values())) * p
        return p
        

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
