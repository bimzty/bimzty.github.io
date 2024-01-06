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

#%% Q3 code
def adams_bashforth(f, a, b, ya, n, method):
    """
    Uses Adams-Bashforth 1-step (Euler) and 2-step methods to approximate the solution to the 
    ODE y' = f(t, y).

    Parameters:
        f (function) : The function that defines the ODE y' = f(t, y).
        a (float) : The initial value of the independent variable t.
        b (float) : The final value of the independent variable t.
        ya (float) : The initial value of the function y(a).
        n (int) : The number of time-steps tp perform.
        method (int) : 1 - Euler, 2 - 2-step method

    Returns:
        t : An array of values from t0 to tn with step size h
        y : An array of solution values at the corresponding values t
    """
    h = (b-a)/n
    t = np.linspace(a, b, n+1)
    y = np.zeros(n+1)

    #In both cases, use 1 step of Euler to obtain y_1
    y[0] = ya
    y[1] = ya+h*f(t[0],y[0])

    #Complete the timestepping
    if method==1: # Apply Euler
        for i in range(1, n):
            y[i+1] = y[i]+h*f(t[i],y[i])
    elif method==2: # Apply Adams-Bashforth 2-step method
        for i in range(1, n):
            y[i + 1] = y[i] + (h / 2) * (3 * f(t[i], y[i]) - f(t[i - 1], y[i - 1]))
    
    return t, y

#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well,
## but these should be written in a separate file
#################################################################

################
#%% Q3 Test
################

# Initialise
a = 0
b = 2
ya = 0.5
n = 40

# Define the ODE and the true solution
f = lambda t, y: y - t**2 + 1
y_true = lambda t: (t + 1)**2 - 0.5*np.exp(t)

# Compute the numerical solutions
t_euler, y_euler = adams_bashforth(f, a, b, ya, n, 1)
t_ab, y_ab = adams_bashforth(f, a, b, ya, n, 2)

print("\n################################")
print("Q3 TEST OUTPUT (last few values of solutions):\n")

# Print the last few points of each solution for comparison
print("  t      True      Euler    Adams-Bashforth")
print("--------------------------------------------")
for i in range(-4, 0):
    print("{:.2f}   {:.6f}   {:.6f}   {:.6f}".format(t_euler[i], 
          y_true(t_euler[i]), y_euler[i], y_ab[i]))

############################################################

