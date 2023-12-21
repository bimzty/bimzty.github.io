"""
MATH2019 CW1 rootfinders module

@author: Taiyuan Zhang
"""


import numpy as np



def bisection(f,a,b,Nmax):
    
    """
    Bisection Method: Returns a numpy array of the 
    sequence of approximations obtained by the bisection method.
    
    Parameters
    ----------
    f : function
        Input function for which the zero is to be found.
    a : real number
        Left side of interval.
    b : real number
        Right side of interval.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    # Start loop
    for i in np.arange(Nmax):
        # Bisect
        p = (a+b)/2
        # Store
        p_array[i] = p
        # Check sign and update
        if f(p)*f(a) > 0: 
            a=p
        else: 
            b=p
    
    # Method finished
    return p_array




def fixedpoint_iteration(g,p0,Nmax):
    """
    Fixed point iteration method: Returns a numpy array of the 
    sequence of approximations obtained by the fixed point iteration method.
    
    Parameters
    ----------
    g : function
        Input function for which the fixed point is to be found.
    p0 : real number
        Initial approximation of fixed point
    Nmax : integer
        Number of iterations to be performed.

    Returns
    -------
    p_array : numpy.ndarray
        Array of shape (Nmax,) containing the sequence of approximations 

    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    # Initial p
    p = p0
    
    # Start loop
    for i in np.arange(Nmax):
        # Iterate
        p = g(p)
        # Store
        p_array[i] = p
        
    # Method finished
    return p_array




def fixedpoint_iteration_stop(g,p0,Nmax,TOL):
    """
    Fixed point iteration method with stopping criterion: 
    Returns a numpy array of the sequence of approximations 
    obtained by the fixed point iteration method 
    using an early stopping criterion.

    Parameters
    ----------
    g : function
        Input function for which the fixed point is to be found.
    p0 : real number
        Initial approximation of fixed point
    Nmax : integer
        Number of iterations to be performed
    TOL : real number
        Tolerance used to stop iterations (once |p-g(p)| <= TOL)

    Returns
    -------
    p_array : numpy.ndarray
        Array containing the sequence of approximations. 
        The shape is at most (Nmax,), but can be smaller 
        (when the stopping criterion is met before reaching Nmax iterations)

    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    # Initial p
    p = p0
    
    # Start loop
    for i in np.arange(Nmax):
        # Iterate
        p = g(p)
        # Store
        p_array[i] = p
        # Stopping criterion
        if abs(p-g(p)) <= TOL:
            p_array = p_array[0:i+1]
            return p_array
        
    # Method finished
    return p_array




def newton_stop(f,dfdx,p0,Nmax,TOL) :
    """
    Newton's method with stopping criterion: 
    Returns a numpy array of the sequence of approximations 
    obtained by Newton's method 
    using an early stopping criterion.

    Parameters
    ----------
    f : function
        Input function for which the zero is to be found.
    dfdx : function
        Input function which is the derivative of f.
    p0 : real number
        Initial approximation of solution
    Nmax : integer
        Number of iterations to be performed
    TOL : real number
        Tolerance used to stop iterations 
        (once subsequent approximations differ <= TOL)

    Returns
    -------
    p_array : numpy.ndarray
        Array containing the sequence of approximations. 
        The shape is at most (Nmax,), but can be smaller 
        (when the stopping criterion is met before reaching Nmax iterations)

    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)

    # algorithm
    p = p0
    p_prev = p0
    for i in np.arange(Nmax):
        p = p - f(p) / dfdx(p)
        # Store
        p_array[i] = p
        # Stopping criterion
        if abs(p-p_prev) <= TOL:
            p_array = p_array[0:i+1]
            return p_array
#        
        p_prev = p

    return p_array


import matplotlib.pyplot as plt 

def plot_convergence(p,f,dfdx,g,p0,Nmax) :
    
    # Fixed-point iteration
    p_array = fixedpoint_iteration(g,p0,Nmax)
    e_array = np.abs(p - p_array)
    n_array = 1 + np.arange(np.shape(p_array)[0])
    
    # Newton iteration
    TOL = 10**(-16)
    p_array2 = newton_stop(f,dfdx,p0,Nmax,TOL)
    e_array2 = np.abs(p - p_array2)
    n_array2 = 1 + np.arange(np.shape(p_array2)[0])
    
    # Preparing figure, using Object-Oriented (OO) style; see: 
    # https://matplotlib.org/stable/tutorials/introductory/quick_start.html
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('n') 
    ax.set_ylabel('|p-p_n|')
    ax.set_title("Convergence behaviour")
    ax.grid(True);
    
    # Plot 
    ax.plot(n_array , e_array , 'o', label='FP iteration',linestyle='--') 
    ax.plot(n_array2, e_array2, 'd', label='Newton',linestyle='-.') 
    ax.legend()

    return fig, ax

    
def optimize_FPmethod(f,c_array,p0,TOL):
    
    # Initialize
    Nmax = 100
    n_array = np.zeros(np.shape(c_array)[0])
    
    # Start loop
    i=0
    for c in c_array:
        # Set g
        g = lambda x: x - c * f(x)
        # Call method
        p_array = fixedpoint_iteration_stop(g,p0,Nmax,TOL)
        # Store output
        n_array[i] = np.shape(p_array)[0]
        # Update i
        i = i + 1

    # Compute optimum
    i_opt = np.argmin(n_array)
    n_opt = n_array[i_opt]
    c_opt = c_array[i_opt]
        
    return c_opt, n_opt