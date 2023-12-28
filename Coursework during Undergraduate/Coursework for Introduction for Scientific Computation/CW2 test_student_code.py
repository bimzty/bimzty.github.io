"""
MATH2019 Coursework 2 main script

@author: Richard Rankin
"""



#%% Question 2

import numpy as np
import systemsolvers as ss

# Initialise 
A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])
b = np.array([[7],[6],[4]])
s = np.array([[5],[20],[10]])
n = 3

# Run find_max
p = ss.find_max(np.hstack((A,b)),s,n,0)
# Print output
print(p)



#%% Question 3

import numpy as np
import systemsolvers as ss

# Initialise 
A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])
b = np.array([[7],[6],[4]])
n = 3

# Run scaled_partial_pivoting
M = ss.scaled_partial_pivoting(A,b,n,1)
# Print output
print(M)

# Run spp_solve
x = ss.spp_solve(A,b,n)
# Print output
print(x)



#%% Question 4 (A1)
import numpy as np
import systemsolvers as ss

# Initialise 
A1 = np.array([[1,-1,2],[1,-2,-1],[2,0.0,2]])
n = 3

# Run PLU
P, L, U = ss.PLU(A1,n)
# Print output
print(P)
print(L)
print(U)


#%% Question 4 (A2)
import numpy as np
import systemsolvers as ss

# Initialise 
A2 = np.array([[1,-1,2],[1,-1,-1],[2,0.0,2]])
n = 3

# Run PLU
P, L, U = ss.PLU(A2,n)
# Print output
print(P)
print(L)
print(U)



#%% Question 5

import numpy as np
import systemsolvers as ss

# Initialise
A = np.array([[4,-1,0.0],[-1,8,-1],[0.0,-1,4]])
b = np.array([[48],[12],[24]])
n = 3
x0 = np.array([[0.0],[0.0],[0.0]])
N = 2

# Run Jacobi
x_approx = ss.Jacobi(A,b,n,x0,N)
# Print output
print(x_approx)



#%% Question 6

import numpy as np
import systemsolvers as ss

# Initialise
A = np.array([[4,-1,0.0],[-1,8,-1],[0.0,-1,4]])
b = np.array([[48],[12],[24]])
n = 3
x0 = np.array([[0.0],[0.0],[0.0]])
N = 10

# Run Jacobi_plot
ss.Jacobi_plot(A,b,n,x0,N)
