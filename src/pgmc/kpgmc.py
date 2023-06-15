import numpy as np
import scipy as sp

import embeddings

class KPGMC:
	def __init__(self, kernel=None, embedding=None):
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

	def fit(self, X, y):
		assert(len(X)==len(y))
		self.K = list(set(y)) # Classes
		self.X = np.array(list(map(self.embedding,X)))
		Ginv = np.array([[0. for j in range(len(self.X))] for i in range(len(self.X))],dtype=float)
		for i in range(len(self.X)):
			for j in range(i,len(self.X)):
				Ginv[i,j] = self.kernel(self.X[i],self.X[j])
				Ginv[j,i] = Ginv[i,j]
		Ginv = sp.linalg.sqrtm(np.linalg.pinv(Ginv))
		self.POVM = []
		for k in self.K:
			self.POVM.append(np.dot(Ginv,np.dot(np.diag([1. if i==k else 0. for i in y]),Ginv)))

	def predict_one(self, w):
		W = np.array([self.kernel(x,w) for x in self.X],dtype=float)
		Wt = np.transpose(W)
		p = [np.dot(Wt,np.dot(self.POVM[i],W)) for i in range(len(self.K))]
		i = np.argmax(p)
		return self.K[i]

	def predict(self, X):
		return np.array(list(map(self.predict_one,np.array(list(map(self.embedding,X))))),dtype=float)


