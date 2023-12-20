import numpy as np
import matplotlib.pyplot as plt
import approximations as approx

####################################################################

def interpolation_errors(a,b,n,P,f):
    """
    (a)
    The error plot shows a smooth decrease in error as the polynomial degree
    is reduced. [0.5] Considering the error bound, important terms are the (p+1)th
    derivative of f and the (p+1)! term. For f(x) = exp(2x), the maximum
    value of the derivative grows like 2^(p+1), but (p+1)! grows faster and
    hence the error decreases. [0.5]

    (b) In this case there is no smooth decrease in error and increasing p does
    not always give a reduction in the error. [0.5] For f(x) = 1/(1+25x^2) over the
    interval [-5,5], the maximum of the (p+1)th derivative of f(x)
    grows extremely quickly with p, hence the bound tells us the error could be very
    large. We see that we do get these very large errors in this case. [0.5] Note,
    this is called the 'Runge Phenomoenon' and f(x) is called the Runge
    function. Careful construction of non-uniformly nodal points can
    solve this problem.
    """ 

    no_eval_points = 2000
    x = np.linspace(a,b,no_eval_points)
    y = f(x) #Evaluate the exact values for comparision

    error_matrix = np.zeros([n])

    for k, p in enumerate(P):
        #Construct the interpolant
        interp = approx.poly_interpolation(a,b,p,no_eval_points,x,f,False)[0]

        #Find maximum errors
        error_matrix[k] = np.max(np.abs(y-interp))


    fig = plt.figure()

    plt.semilogy(P,error_matrix,'b--*')
    plt.xlabel('p')
    plt.ylabel('$\max_{x \in [a,b]} |f(x) - p_p(x)|$')
    plt.title('Polynomial Interpolation Error Convergence')

    return error_matrix, fig

####################################################################

def derivative_errors(x,P,m,H,n,f,fdiff):
    """
    (a)
    For f(x) = exp(2x), we see O(h^p) convergence in all cases [0.5].
    This is expected from the theory because differentiating the 
    polynomial interpolant error bound and setting the evaluation 
    point to be the central node yields O(h^p) convergence [0.25]. 
    For the highest value of p, the error flattens off for small h, 
    this is due to roundoff error [0.25]

    (b) 
    For the piecewise function, we see only O(h) convergence for each 
    value of p [0.5]. 
    The theory for polynomial interpolation errors fails in this case, 
    as f \in C^1[-1,1] so we cannot realy tell what order we should be 
    expecting [0.5]. 
    In fact, by construction of the difference quotients by hand, 
    we see that we should expect the O(h) convergence for this
    function. More general convergence results exist for non-smooth
    functions.
    """

    E = np.zeros([m,n])

    fig = plt.figure()
    legend_entries = []    

    for k, p in enumerate(P):
        for j, h in enumerate(H):
            pos = p/2
        
            deriv_approx = approx.approximate_derivative(x,p,h,pos,f)
        
            E[k,j] = np.abs(fdiff(x)-deriv_approx)
        
        legend_entries.append("$p = $"+ str(P[k]))
        plt.loglog(H,E[k,:],":s")
        plt.title("Convergence of Derivative Errors")
        plt.xlabel("$h$")
        plt.ylabel("$|f'(x)-p'_{p,h}|$")
        plt.legend(legend_entries)

    return E, fig

print("\nAny outputs above this line are due to importing approximations.py.\n")

################
## Test Code ##
################

################
#%% Q3 Test
################

# Initialise
n = 5
P = np.arange(1,n+1)
a = -1
b = 1
f = lambda x: 1/(x+2)

#Run the function
error_matrix, fig = interpolation_errors(a,b,n,P,f)

print("\n################################")
print('Q3 TEST OUTPUT:\n')

print("error_matrix = \n")
print(error_matrix)

################
#%% Q7 Test
################

#Initialise
P = np.array([2,4,6])
H = np.array([1/4,1/8,1/16])
x = 0
f = lambda x: 1/(x+2)
fdiff = lambda x: -1/((x+2)**2)

#Run the function
E, fig = derivative_errors(x,P,3,H,3,f,fdiff)

print("\n################################")
print("Q7 TEST OUTPUT:\n")

print("E = \n")
print(E)
plt.show()

