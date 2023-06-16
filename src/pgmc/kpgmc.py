import numpy as np
import scipy as sp
import copy
import sklearn as sk

from pgmc import embeddings

class KPGMC:
    def __init__(self, kernel=None, embedding=None, class_weight=None, kernel_parameters=None):
        self.class_weight = class_weight
        self.kernel_parameters = kernel_parameters

        if kernel==None:
            self.kernel = np.dot
        else:
            self.kernel = kernel

        if embedding==None:
            self.embedding = embeddings.normalize
        elif embedding.lower() == "normal":
            self.embedding = embeddings.normalize
        elif embedding.lower() == "stereo":
            self.embedding = embeddings.stereo
        elif embedding.lower() == "orthogonal":
            self.embedding = embeddings.orthogonalize
        else:
            self.embedding = embedding

        self.POVM = []
        self.K = []
        self.X = []

    def _adjust_class_weight(self):
        # Adjust class weights
        if type(self.class_weight) == dict:
            pass
        elif type(self.class_weight) == list:
            self.class_weight = {self.K[i] : self.class_weight[i] for i in range(len(self.K))}
        elif self.class_weight == "auto":
            self.class_weight = {k: min(max(2/len(self.K)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.K}
        elif self.class_weight == "optimize":
            self._optimize_class_weight()
        else:
            self.class_weight = {k:1. for k in self.K}

    def _compute_matrices_pi(self):
        self.Kernel_Matrix = np.array([[0. for j in range(len(self.X))] for i in range(len(self.X))],dtype=float)
        for i in range(len(self.X)):
            for j in range(i,len(self.X)):
                self.Kernel_Matrix[i,j] = self.kernel(self.X[i],self.X[j])
                self.Kernel_Matrix[j,i] = self.Kernel_Matrix[i,j]
        Ginv = sp.linalg.sqrtm(np.linalg.pinv(self.Kernel_Matrix))
        self.POVM = []                                       
        for k in self.K:
            self.POVM.append(np.real(np.dot(Ginv,np.dot(np.diag([1. if i==k else 0. for i in self.y]),Ginv))))

    def _optimize_class_weight(self):
        self.class_weight = {k: min(max(2/len(self.K)- list(self.y).count(k)/len(self.y),0.15),0.85) for k in self.K}
        w = np.array([0.]*len(self.K),dtype=float)
        for i in range(len(self.y)):
            w[self.y[i]] += np.dot(np.dot(self.Kernel_Matrix[i].transpose(),self.POVM[self.y[i]]),self.Kernel_Matrix[i])
        for i in range(len(self.K)):
            w[i] /= list(self.y).count(self.K[i])
        def f_(x):
            score = np.dot(x,w)
            return 1.-score
        default = f_(np.array(list(self.class_weight.values())))
        res = sp.optimize.minimize(f_,np.array(list(self.class_weight.values())),bounds=[(0,1)]*len(self.K),method="Nelder-Mead")
        self.class_weight = {self.K[i]:res.x[i] for i in range(len(self.K))}
        

    def fit(self, X, y):
        assert(len(X)==len(y))
        self.K = list(set(y)) # Classes
        self.X = np.array(list(map(self.embedding,X)))
        self.y = copy.deepcopy(y)

        self._compute_matrices_pi()
        self._adjust_class_weight()
        
        

    def predict_one(self, w):
        W = np.array([self.kernel(x,w) for x in self.X],dtype=float)
        Wt = np.transpose(W)
        p = [np.dot(Wt,self.class_weight[self.K[i]]*np.dot(self.POVM[i],W)) for i in range(len(self.K))]
        i = np.argmax(p)
        return self.K[i]

    def predict(self, X):
        return np.array(list(map(self.predict_one,np.array(list(map(self.embedding,X))))),dtype=float)


