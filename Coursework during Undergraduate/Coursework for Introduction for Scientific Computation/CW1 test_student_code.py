"""
MATH2019 CW1 main script

@author: Kris van der Zee (lecturer)
"""



#%% Question 2

import numpy as np
import matplotlib.pyplot as plt
import rootfinders as rf
 
# Initialise 
f = lambda x: x**3 + x**2 - 2*x - 2 
a = 1
b = 2
Nmax = 5

# Run bisection
p_array = rf.bisection(f,a,b,Nmax)

# Print output
print(p_array)



#%% Question 3

import numpy as np
import matplotlib.pyplot as plt
import rootfinders as rf
 
# Print help description
help(rf.fixedpoint_iteration)

# Initialise 
g = lambda x: 1 - 1/2 * x**2
p0 = 1
Nmax = 5

# Run method
p_array = rf.fixedpoint_iteration(g,p0,Nmax)

# Print output
print(p_array)


