from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code


# parameters
name = '1'

# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]
# print(len(X))
# print(len(Y))
# print(X)
# print(Y)
# Carry out training.
# ## TODO ###
#
# Define the predict_linearSVM(x) function, which uses the trained parameters
# predict_linearSVM(x) should return the score that the classifier assigns to point x
#   e.g. for linear classification, this means w^T x, not sign(w^T x)
# ## TODO ###

def getWeights(X,Y,max_epochs,lam):
    t=0
    w=np.array([[0]*len(X[0])])
    w_0=0
    while max_epochs>0:
        max_epochs-=1
        for i in range(len(X)):
            t+=1
            step=1/(t*lam)
            if np.dot(np.array([X[i]]),w.T)[0][0]<1:
                w=(1-step*lam)*w+step*Y[i][0]*np.array([X[i]])
                w_0=(1-step*lam)*w_0+step*Y[i][0]
            else:
                w=(1-step*lam)*w
                w_0=(1-step*lam)*w_0
    return (w,w_0)

for i in range(1,-11,-1):
    w,w_0=getWeights(X,Y,10,2**(i))
    print(i,np.dot(w,w.T)[0][0])
# print(w,w_0)
def predict_linearSVM(x):
    return np.dot(np.array([x]),w.T)[0][0]+w_0
    # pass

# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()
