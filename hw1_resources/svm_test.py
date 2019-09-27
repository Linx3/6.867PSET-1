from numpy import *
from numpy import linalg as LA
from plotBoundary import *
import pylab as pl
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

# import your SVM training code


# parameters
name = '1'

print('======Training======')
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
# print(train)
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()
gamma=0.05
# Carry out training, primal and/or dual
### TODO ###

# Define the predictSVM(x) function, which uses the trained parameters
# predictSVM(x) should return the score that the classifier assigns to point x
#   e.g. for linear classification, this means w^T x, not sign(w^T x)
### TODO ###
# print(Y)
# X=[[2,2],[2,3],[0,-1],[-3,-2]]
# Y=[[1],[1],[-1],[-1]]
def get_matrix_linear():
    K = zeros((len(X),len(X)))
    for i in range(len(K)):
        x1=np.array([X[i]])
        for j in range(len(K)):
            x2=np.array([X[j]])
            K[i][j]=np.dot(x1,x2.T)[0][0]*Y[i][0]*Y[j][0]
    # print(K)
    K*=0.5
    # print(K)
    return matrix(K,tc="d")

def get_matrix_gaussian():
    K = zeros((len(X),len(X)))
    for i in range(len(K)):
        x1=np.array([X[i]])
        for j in range(len(K)):
            x2=np.array([X[j]])
            K[i][j]=math.exp(-gamma*np.dot((x1-x2),(x1-x2).T)[0][0])
    # print(K)
    K*=0.5
    # print(K)
    return matrix(K,tc="d")

def solve_aplhas(C):
    P=get_matrix_gaussian()
    q=matrix([-1.0]*len(X))
    G1=np.array(Y).T*1.0
    G2=-1*G1
    G3=-np.identity(len(X))
    G4=np.identity(len(X))
    # print(G1,G2,G3,G4)
    G=matrix(np.concatenate((G1,G2,G3,G4),axis=0))
    h=matrix([0.0,0.0]+len(X)*[0.0]+len(X)*[C*1.0])
    # print(P,q,G,h)
    sol = solvers.qp(P,q,G,h)
    return sol["x"]

C=100
alpha=solve_aplhas(C)
# for i in alpha:
#     print(i)
count=0
for i in alpha:
    print(i)
    if i>10**(-4) and i<C:
        count+=1
print("support vectors",count,(len(X)))
# print(1/0)
# weight=np.array([[0.0,0.0]])
# for i in range(len(X)):
    # print(weight)
    # print(alphas[i]*Y[i][0]*(np.array([X[i]])))
    # weight+=alphas[i]*Y[i][0]*(np.array([X[i]]))
# print("this weight")
# print(weight)
# print("1/this weight",1/LA.norm(weight))
# def predictSVM(x):
#     return np.dot(weight,np.array([x]).T)[0][0]
def predictSVM(x):
    print(x)
    result=0
    x=np.array([x])
    for i in range(len(X)):
        x_i=np.array([X[i]])
        product=math.exp(-gamma*np.dot(x-x_i,(x-x_i).T)[0][0])
        result+=alpha[i]*Y[i][0]*product
    return result
# plot training results
# print(predictSVM([1,2]))
# plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')

# print('======Validation======')
# load data from csv files

# training_count=0
# training_wrong=0
# for i in range(len(X)):
#     training_count+=1
#     classify=predictSVM(X[i])
#     if classify*Y[i][0]<0:
#         training_wrong+=1
#
#
# validate = loadtxt('data/data'+name+'_validate.csv')
# X = validate[:, 0:2]
# Y = validate[:, 2:3]
# validation_count=0
# validation_wrong=0
# for i in range(len(X)):
#     validation_count+=1
#     classify=predictSVM(X[i])
#     if classify*Y[i][0]<0:
#         validation_wrong+=1
# print("result")
# print(training_count,training_wrong,validation_count,validation_wrong)

# plot validation results
# plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
