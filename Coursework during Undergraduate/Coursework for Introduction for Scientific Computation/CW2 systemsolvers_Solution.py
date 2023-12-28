"""
MATH2033 Coursework 2 systemsolvers module

@author: Taiyuan Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import warmup_solution as ws



def find_max(M,s,n,i):
    
    """
    Returns the smallest integer p such that p is at least i and |M[p,i]|/s[p]
    is the maximum over j in {i,i+1,...,n-1} of |M[j,i]|/s[j].
    
    Parameters
    ----------
    M : numpy.ndarray of shape (n,n+1)
        array representing an n by n+1 matrix.
    s : numpy.ndarray of shape (n,1)
        array with no nonpositive elements.
    n : integer
        integer that is at least 2.
    i : integer
        nonnegative integer that is at most n-2.
    
    Returns
    -------
    p : integer
        the smallest integer p such that p is at least i and |M[p,i]|/s[p] is
        the maximum over j in {i,i+1,...,n-1} of |M[j,i]|/s[j].
    """
    
    # Continue here:...
    m=0
    p=i
    for j in range(i,n):
        v=abs(M[j,i])/s[j]
        if v>m:
            m=v
            p=j
            
    return p



def scaled_partial_pivoting(A,b,n,c):
    
    """
    Returns an array representing the augmented matrix M arrived at by
    starting from the augmented matrix [A b] and performing forward
    elimination with scaled partial pivoting until all of the entries below
    the main diagonal in the first c columns are 0.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the square matrix A.
    b : numpy.ndarray of shape (n,1)
        array representing the column vector b.
    n : integer
        integer that is at least 2.
    c : integer
        positive integer that is at most n-1.
    
    Returns
    -------
    M : numpy.ndarray of shape (n,n+1)
        array representing the augmented matrix M.
    """
    
    s=np.amax(np.abs(A),1)
    
    # Continue here:...
    M=np.hstack((A,b))
    
    for i in range(c):
        p=find_max(M,s,n,i)
        M[[i,p],:]=M[[p,i],:]
        s[[i,p]]=s[[p,i]]
        for j in range(i+1,n):
            m=M[j,i]/M[i,i]
            M[j,i]=0
            M[j,i+1:n+1]=M[j,i+1:n+1]-m*M[i,i+1:n+1]
    
    return M



def spp_solve(A,b,n):
    
    """
    Returns the solution x to Ax=b computed using forward elimination with
    partial pivoting followed by backward substitution.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the matrix A in the linear system Ax=b.
    b : numpy.ndarray of shape (n,1)
        array representing the vector b in the linear system Ax=b.
    n : integer
        integer that is at least 2.
    
    Returns
    -------
    x : numpy.ndarray of shape (n,1)
        array representing the solution x to Ax=b.
    """
    
    # Continue here:...
    M=scaled_partial_pivoting(A,b,n,n-1)
    x=ws.backward_substitution(M,n)
    
    return x



def PLU(A,n):
    
    """
    Returns arrays representing a permutation matrix P, a lower triangular
    matrix L and an upper triangular matrix U such that A=PLU.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the matrix A.
    n : integer
        integer that is at least 2.
    
    Returns
    -------
    P : numpy.ndarray of shape (n,n)
        array representing the permutation matrix P.
    L : numpy.ndarray of shape (n,n)
        array representing the lower tirangular matrix L.
    U : numpy.ndarray of shape (n,n)
        array representing the upper triangular matrix U.
    """
    
    # Set P=I
    P=np.identity(n)
    # Set L to be a zero matrix
    L=np.zeros([n,n])
    # Set U=A
    U=A.copy()
    
    # Continue here:...
    for i in range(n-1):
        k=0
        while abs(U[i+k,i])<=10**(-15):
            k+=1
        if k!=0:
            P[[i,i+k],:]=P[[i+k,i],:]
            L[[i,i+k],:]=L[[i+k,i],:]
            U[[i,i+k],:]=U[[i+k,i],:]
        for j in range(i+1,n):
            L[j,i]=U[j,i]/U[i,i]
            U[j,i]=0
            U[j,i+1:n]=U[j,i+1:n]-L[j,i]*U[i,i+1:n]
    P=np.transpose(P)
    L=L+np.identity(n)
    return P, L, U



def Jacobi(A,b,n,x0,N):
    
    """
    Returns an array of approximations to the solution of Ax=b obtained using
    the Jacobi method.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the matrix A
    b : numpy.ndarray of shape (n,1)
        array representing the vector b
    n : integer
        integer that is at least 2.
    x0 : numpy.ndarray of shape (n,1)
        array representing the initial approximation x0.
    numits : integer
        the number of iterations to be performed.
    
    Returns
    -------
    x_approx : numpy.ndarray of shape (n,N+1)
        array whose column 0 is x0 and, for i=1,2,...,N, whose column i is the
        approximation to the solution of Ax=b obtained after performing i
        iterations of the Jacobi method starting from x0.
    """
    
    # Continue here:...
    x_approx=np.zeros([n,N+1])
    for j in range(n):
        x_approx[j,0]=x0[j]
    for i in range(N):
        for j in range(n):
            x_approx[j,i+1]=(b[j]-np.dot(A[j,0:j],x_approx[0:j,i])-np.dot(A[j,j+1:n],x_approx[j+1:n,i]))/A[j,j]
    return x_approx



def Jacobi_plot(A,b,n,x0,N):

    # Create array of k values
    k_array = np.arange(N+1)
    # Prepare figure
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.grid(True)
    
    # Continue here:...
    x = ws.no_pivoting_solve(A,b,n)
    x_approx = Jacobi(A,b,n,x0,N)
    error_i = np.zeros(N+1)
    error_2 = np.zeros(N+1)
    for i in range(N+1):
        v=x[0:n,0]-x_approx[0:n,i]
        error_i[i]=np.linalg.norm(v,np.inf)
        error_2[i]=np.linalg.norm(v,2)
    
    # Plot
    ax.plot(k_array,error_i,"s",label="$||x-x^{(k)}||_\infty$")
    ax.plot(k_array,error_2,"o",label="$||x-x^{(k)}||_2$")
    # Add legend
    ax.legend()
    return fig, ax
