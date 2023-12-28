#################################################################
## Functions to carry out numerical solution of first order IVPS
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import a non-standard modules, ask Ed if that 
## - is acceptable
#################################################################
import numpy as np

#################################################################
## Functions to be completed by student
#################################################################

#%% Q4a code
def adams_bashforth_2(f, a, b, alpha, beta, n, method):
    """
    Uses Adams-Bashforth 1-step (Euler) and 2-step methods to approximate the solution to the 
    second order IVP y'' = f(t, y, y'), with y(a) = alpha and y'(b) = beta. The ODE
    is initially converted into a system of first order IVPs.

    Parameters:
        f (function) : The function that defines the ODE y'' = f(t, y, y').
        a (float) : The initial value of the independent variable t.
        b (float) : The final value of the independent variable t.
        alpha (float) : The initial value of the function y(a).
        beta (float): The initial value y'(a)
        n (int) : The number of time-steps tp perform.
        method (int) : 1 - Euler, 2 - 2-step method

    Returns:
        t : An array of values from t0 to tn with step size h
        y : An array of solution values at the corresponding values t
    """
    h = (b-a)/n
    t = np.linspace(a, b, n+1)
    y = np.zeros([2,n+1])

    #Construct the first order system    
    f_system = lambda t,y: np.array([y[1],f(t,y[0],y[1])])

    #In both cases, use 1 step of Euler to obtain y_1
    y[:,0] = np.array([alpha,beta])
    y[:,1] = y[:,0]+h*f_system(t[0],y[:,0])

    #Complete the time stepping
    if method==1: # Apply Euler
        for i in range(1, n):
            y[:,i+1] = y[:,i]+h*f_system(t[i],y[:,i])
    elif method==2: # Apply Adams-Bashforth 2-step method
        for i in range(1, n):
            y[:,i+1] = y[:,i] + (h / 2) * (3 * f_system(t[i],y[:,i]) - f_system(t[i-1],y[:,i-1]))

    return t, y[0,:]

#%% Q4b code
def compute_ode_errors(n_vals,no_n,a,b,alpha,beta,f,true_y):
    """
    The results indicate that we are achieving O(h)
    convergence for the Euler method, as the error
    reduces by half as h is halved.[0.5] This agrees with
    the result proved in lectures.[0.5] 
    For the 2-step Adams-Bashforth method we see O(h^2)
    convergence as halving h leads to h reducing by 1/4 [0.5]
    Using the Euler method for the first step does not impact
    the O(h^2) convergence as the local error for one step
    of Euler is O(h^2). [0.5]
    """
    errors_y = np.zeros([2,no_n])

    true_val = true_y(b)

    for j, n in enumerate(n_vals):
        for k in range(2):
          t, y = adams_bashforth_2(f, a, b, alpha, beta, n, k+1)
          errors_y[k,j] = np.abs(true_val-y[-1])

    return errors_y

#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well,
## but these should be written in a separate file
#################################################################

# Define the second-order ODE
f = lambda t,y0,y1: (2 + np.exp(-t))*np.cos(t)-y0-2*y1
true_y = lambda t: np.exp(-t)- np.exp(-t)*np.cos(t) + np.sin(t)

a = 0
b = 1
alpha = 0
beta = 1

################
#%% Q4a Test
################

n = 40

# Compute the numerical solutions
t_euler, y_euler = adams_bashforth_2(f, a, b, alpha, beta, n, 1)
t_ab, y_ab = adams_bashforth_2(f, a, b, alpha, beta, n, 2)

print("\n################################")
print("Q4a TEST OUTPUT (last few values of solutions):\n")

# Print the last few points of each solution for comparison
print("  t      True      Euler    Adams-Bashforth")
print("--------------------------------------------")
for i in range(-4, 0):
    print("{:.2f}   {:.6f}   {:.6f}   {:.6f}".format(t_euler[i], 
          true_y(t_euler[i]), y_euler[i], y_ab[i]))
    


################
#%% Q4b Test
################

no_n = 6
n_vals = 4*2**np.arange(no_n)

errors_y = compute_ode_errors(n_vals,no_n,a,b,alpha,beta,f,true_y)

print("\n################################")
print("Q4b TEST OUTPUT:\n")

print("errors_y = \n")
print(errors_y)

############################################################

