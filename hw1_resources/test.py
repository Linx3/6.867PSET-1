from cvxopt import matrix
P = matrix([[1.0,0.0],[0.0,0.0]])
q = matrix([3.0,4.0])
G = matrix([[-1.0,0.0,-1.0,2.0,3.0],[0.0,-1.0,-3.0,5.0,4.0]])
h = matrix([0.0,0.0,-15.0,100.0,80.0])

from cvxopt import solvers
sol = solvers.qp(P,q,G,h)
# print(type(sol))
print(sol)
