from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
import math
# import your LR training code


# parameters
name = '1'

# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# print(len(train))
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
# epochs = 1000;
epochs=1
lmbda = .02;
gamma = 2**2   ;

n=len(X)
K = zeros((n,n))
def Gaussian_matrix(gamma):
    for i in range(len(K)):
        for j in range(len(K)):
            x=np.array([X[i]])
            y=np.array([X[j]])
            norm=np.dot((x-y),(x-y).T)[0][0]
            K[i][j]=math.exp(-gamma*norm)

### TODO: Compute the kernel matrix ###

### TODO: Implement train_gaussianSVM ###
Gaussian_matrix(gamma)

def train_gaussianSVM(X,Y,K,lam,max_epochs):
    t=0
    a=[0]*len(X)
    while max_epochs>0:
        for i in range(len(X)):
            t+=1
            step=1/(t*lam)
            predict=0
            for j in range(len(X)):
                predict+=a[j]*Y[j][0]*K[j][i]
            predict*=Y[i][0]
            if predict<1:
                a[i]=(1-step*lam)*a[i]+step
            else:
                a[i]=(1-step*lam)*a[i]
        max_epochs-=1
    return a


alpha = train_gaussianSVM(X, Y, K, lmbda, epochs)
total=0
for i in (alpha):
    if i>10**(-4):
        total+=1
print("vvvvv")
print(total,len(X))


def predict_gaussianSVM(x):
    print(x)
    result=0
    x=np.array([x])
    for i in range(len(X)):
        x_i=np.array([X[i]])
        product=math.exp(-gamma*np.dot(x-x_i,(x-x_i).T)[0][0])
        result+=alpha[i]*Y[i][0]*product
    return result

# Define the predict_gaussianSVM(x) function, which uses the trained parameters alpha
# predict_gaussianSVM(x) should return the score that the classifier assigns to point x
#   e.g. for linear classification, this means w^T x, not sign(w^T x)
### TODO ###

# plot training results
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
pl.show()
