import numpy as np
import matplotlib.pyplot as plt

### Function to carry out composite trapezium rule with n strips

def composite_trapezium(a,b,n,f):
    """
    To approximate the definite integral of the function f(x) over the interval [a,b]
    using the composite trapezoidal rule with n subintervals.   

    Parameters:
    -----------
    f (function): The function to integrate
    a (float): The lower limit of integration
    b (float): The upper limit of integration
    n (Integer): The number of subintervals to use in the composite trapezoidal rule
    Returns
    -------
    approx (float): The approximation of the integral      
    """
    
    x = np.linspace(a,b,n+1) #Construct the quadrature points
    h = (b-a)/n

    #Construct the quadrature weights: 
    #These are the coefficient w_i of f(x_i) in the summation
    weights = h*np.ones(n+1) 
    weights[[0,-1]] = h/2

    approx_integral = np.sum(f(x)*weights)

    return approx_integral


### Function to carry out Romberg integration. level 1 = composite trapezium with n strips

def romberg_integration(a,b,n,f,level):
    
    """
    Computes the definite integral of a function f over the interval [a,b] 
    using Romberg integration with a given number of subintervals n and 
    specified level of refinement.
    
    Parameters
    ----------
    a (float): The lower limit of the integration interval
    b (float): The upper limit of the integration interval
    n (integer): The number of subintervals to use in the initial computation 
               of the composite trapezoidal rule
    f (callable function): The function to integrate
    level(integer): The number of levels to use in the Romberg integration
         
    Returns
    -------
    Values (numpy.ndarray of shape (level, level): the computed value of the definite integral    
    """
   
    values = np.zeros([level,level])
    #Compute level 1 solution with composite trapezium
    
    for k in range(level):

        values[0,k] = composite_trapezium(a,b,n*(2**k),f)

    #Now perform the Richardson extrapolation

    for k in range(1,level):
        alpha = (4**k)/(4**k-1)
        beta = -1/(4**k-1)

        values[k,:level-k] = alpha*values[k-1,1:level-k+1]+beta*values[k-1,:level-k]

            
    return values[level-1,0]

### Test function to make sure we have O(h**(2*level)) convergence
def compute_errors(N,no_n,levels,no_levels,f,a,b,true_val):
    """
    Straight lines on a log-log plot indicate we have have O(n^(-p)
    convergence for some p. In fact, the results show that 
    p = 2*level in each case. This is equivalent to O(h^(2*level)) 
    convergence. [1] 
    This matches entirely with the Richardson iteration theory
    using the fact that the trapezium rule errors depend only on the 
    even powers of h. [1]      
    """
    error_matrix = np.zeros([no_levels,no_n])
    legend_vals = []

    fig = plt.figure()
    for l,level in enumerate(levels):
        legend_vals.append("level = "+str(l+1))
        for k,n in enumerate(N):
            error_matrix[l,k] = np.abs(true_val-romberg_integration(a,b,n,f,level))

        plt.loglog(N,error_matrix[l,:],":s")

    plt.title('Convergence of Romberg Integration')
    plt.xlabel('$n$')
    plt.ylabel(r'$|$Error$|$')
    plt.legend(legend_vals)

    return error_matrix,fig

################
## Test Code ##
################


################
#%% Q1 Test
################

# Initialise
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 10

#Run the function
approx_integral = composite_trapezium(a,b,n,f)

print("\n################################")
print("Q1 TEST OUTPUT:\n")
print("approx_integral =\n")
print(approx_integral)
print("")

################
#%% Q2a Test
################

# Initialise
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 2

print("\n################################")
print("Q2a TEST OUTPUT:\n")
print("")

# Test code
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 2

for level in range(1,5):
    #Run test 
    integral_approx = romberg_integration(a,b,n,f,level)
    print("level = " + str(level)+ ", integral_approx = " + str(integral_approx))



#%% Q2b
################
N = [1,2,4,8]
levels = [1,2,3,4]
true_val = 2.0
error_matrix, fig = compute_errors(N,4,levels,4,f,a,b,true_val)

print("\n################################")
print("Q2b TEST OUTPUT:\n")
print("Error =\n")
print(error_matrix)
#plt.show()
#print("")


