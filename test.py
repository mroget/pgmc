from src.pgmc import kpgmc
from src.pgmc import embeddings

from numba import jit

# Maths and data management
import math
from math import pi
import numpy as np
import scipy as sp
import pandas as pd
import pickle
import time
from tqdm import tqdm


# ML toolkit
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier

from sklearn.model_selection import RandomizedSearchCV

def get_mnist(features=5, nb_classes=2, path="mnist.pkl"):
    """
    Load MNIST, eliminate some classes and apply pca.
    """
    pca = PCA(n_components=features)
    (X,y) = pickle.load(open(path,"rb"))
    X = np.array([X[i] for i in range(len(X)) if y[i] in list(range(nb_classes))])
    y = np.array([y[i] for i in range(len(y)) if y[i] in list(range(nb_classes))])
    if len(X[0]) > features:
        X = pca.fit_transform(X)
    return X,y

def load_imagenet(features=12288, nb_classes=200, size=10000):
    from datasets import load_dataset
    ds = load_dataset("zh-plus/tiny-imagenet").with_format("np")
    X = []
    y = []

    l = list(ds["valid"])
    K = set([i["label"] for i in l])
    classes = {k:[] for k in K}

    for img in tqdm(l):
        if len(np.reshape(img["image"],(-1,))) == 12288:
            classes[img["label"]].append(np.reshape(img["image"],(-1,)))

    for i in range(len(list(classes.values())[0])):
        for k in classes:
            if i >= len(classes[k]):
                break
            X.append(classes[k][i])
            y.append(k)
            if len(X)>=size:
                break
        if len(X)>=size:
                break
    X = np.array(X)
    y = np.array(y)
    
    pca = PCA(n_components=features)
    X = np.array([X[i] for i in range(len(X)) if y[i] in list(range(nb_classes))])
    y = np.array([y[i] for i in range(len(y)) if y[i] in list(range(nb_classes))])
    if len(X[0]) > features:
        X = pca.fit_transform(X)
    return X,y

def metrics(y_true,y_pred,average="binary",silent=False):
    """
    Compute and return a lot of metrics given prediction and ground truths.
    """
    if len(list(set(y_pred))) > 2 and average == "binary":
        average = "micro"
    accuracy = sk.metrics.accuracy_score(y_true,y_pred)
    precision = sk.metrics.precision_score(y_true,y_pred,average=average,zero_division=0)
    recall = sk.metrics.recall_score(y_true,y_pred,average=average,zero_division=0)
    ba = sk.metrics.balanced_accuracy_score(y_true, y_pred)
    mse = sk.metrics.mean_squared_error(y_true, y_pred)
    confusion = sk.metrics.confusion_matrix(y_true,y_pred)
    fmeas = sk.metrics.f1_score(y_true,y_pred,average=average,zero_division=0)
    if not silent:
        print("Accuracy : ",accuracy)
        print("Precision : ",precision)
        print("Recall : ",recall)
        print("BA : ",ba)
        print("MSE : ",mse)
        print("F-measure : ",fmeas)
        print("Confusion matrix : \n",confusion)
    return [accuracy,precision,recall,ba,mse,fmeas,confusion]


def imbalance(X,y,ratio):
    """
    Articificially imbalance a dataset.
    """
    X0 = np.array([[X[i],y[i]] for i in range(len(y)) if y[i] == 0],dtype=object)
    X1 = np.array([[X[i],y[i]] for i in range(len(y)) if y[i] == 1],dtype=object)
    current = len(X1)/(len(X0)+len(X1))
    if current > ratio:
        desired_len = int(len(X0)*ratio/(1-ratio))
        c1 = np.random.choice(list(range(len(X1))),desired_len,replace=False)
        X1 = X1[c1]
    else:
        desired_len = int(len(X1)*(1-ratio)/ratio)
        c0 = np.random.choice(list(range(len(X0))),desired_len,replace=False)
        X0 = X0[c0]
    data = np.concatenate([X0,X1])
    X_ = np.array(list(data[:,0]))
    y_ = np.array(list(data[:,1]))
    return X_,y_


class Task:
    def __init__(self, repeat=5):
        self.datasets = []
        self.clf = []
        self.repeat=repeat

    def todo(self):
        ret = []
        for data_name,X,y in self.datasets:
            for name_clf,clf in self.clf:
                ret.append([data_name,X,y,name_clf,clf,self.repeat])
        return ret

    def add_data(self,name,X,y):
        self.datasets.append((name,X,y))

    def add_clf(self,name,clf):
        self.clf.append((name,clf))

    def run_aux(self,clf,X_train,y_train,X_test,y_test):
        T = time.time()
        clf.fit(X_train,y_train)
        T = time.time() - T
        y_pred = clf.predict(X_test)
        l = metrics(y_test,y_pred,silent=True,average="binary" if len(set(y_test))==2 else "micro")
        return [len(y_train),len(X_train[0])]+l[0:6]+[T]

    def run(self):
        res = []
        for name_data,X,y,name_clf,clf,repeat in tqdm(self.todo()):
                rs = sk.model_selection.ShuffleSplit(n_splits=repeat, test_size=0.3, random_state=0)
                for i, (train_index, test_index) in tqdm(enumerate(rs.split(X)),total=repeat,desc=f"{repeat}-fold CrossValidation of {name_clf} on {name_data}",leave=False):
                    res.append(self.run_aux(clf(),X[train_index],y[train_index],X[test_index],y[test_index])+[name_clf,name_data])
        return pd.DataFrame(res,columns=["size","features","acc","precision","recall","ba","mse","f1","time","clf","data"])

@jit
def rbf_kernel(x,y,c):
    return np.exp(-c*np.linalg.norm(x-y)**2)


#rbf_kernel(np.array([1.,1.]), np.array([1.,1.]))


task = Task(repeat=5) # 5-fold crossvalidation

## DATASETS
#X,y = get_mnist(40,10,path="mnist1d.pkl")
#X,y = get_mnist(40,10,path="../mnist.pkl")
#X,y = imbalance(*get_mnist(40,2,path="../mnist1d.pkl"),0.1) # i features 2 classes
#task.add_data("MNIST-1D 0.9|0.1",X,y)
#X,y = X[::7],y[::7]
#task.add_data("MNIST-1D 0.5|0.5",X,y)

X,y = load_imagenet(features=100, nb_classes=200, size=1000)

task.add_data("Mininet",X,y)

## CLASSIFIERS
task.add_clf("KPGMC rbf",lambda :kpgmc.KPGMC(kernel=rbf_kernel, kernel_parameter=2, class_weight="auto", device="cpu"))
task.add_clf("KPGMC ortho",lambda :kpgmc.KPGMC(embedding="orthogonal",class_weight="auto", device="cpu"))
task.add_clf("SVM rbf",lambda :SVC())
data = task.run()

print(data.groupby(["data","clf"]).mean(numeric_only=False))
"""

param_distributions = {"kernel_parameter": np.linspace(0.01,1.,10)}

clf = kpgmc.KPGMC(kernel=rbf_kernel, kernel_parameter=1., class_weight="auto", device="cpu")

search = RandomizedSearchCV(clf, param_distributions,n_iter=10).fit(X, y)

print(search.best_params_)"""