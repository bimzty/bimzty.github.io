import numpy as np
import matplotlib.pyplot as plt
import lagrange_polynomials as lp

####################################################################

def poly_interpolation(a,b,p,n,x,f,produce_fig):
    """
    Finds the polynomial interpolant of order p of a function f
    using uniformly spaced points over the interval [a,b]
    
    Optionally draws a graph of this interpolant (and the original f)

    Parameters
    ----------
    a, b : floats
          limits on which the interpolant is based (a < b assumed)
    p : int
          polynomial degree to use (assumed positive)
    n : int
          number of points at which to evaluate the interpolant
    x : numpy.ndarray of shape (n,)
          points at which to evaluate the interpolant
    f : function of one variable
          function to find the interpolant of
    produce_fig : bool
          True - plot the interpolant, False - don't

    Returns
    -------
    interpolant : numpy.ndarray of shape (n,)
        The interpolant evaluated at all points in in x
    fig :: matplotlib.matplotlib.figure.Figure/None
        Plot of interpolant

    Examples
    --------
    >>> poly_interpolation(0,1,2,2,np.array([1,2]),lambda x: x**3,True)
    """
    
    #Set up nodal points
    xhat = np.linspace(a,b,p+1)

    #Compute Lagrange interpolating polynomials
    L = lp.lagrange_poly(p,xhat,n,x,1.0e-10)[0]

    #Evaluate f at the nodal points
    y = f(xhat)

    #Compute the interpolant - using a matrix product
    interpolant = np.matmul(y,L)

    if produce_fig:
        fig = plt.figure()
        plt.plot(x,f(x),'r-')
        plt.plot(x,interpolant,'b-')
        plt.legend(['$f(x)$','$p_{'+str(p)+'}(x)$'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Polynomial Interpolation of degree '+str(p))
    else:   
        fig = None

    return interpolant, fig

####################################################################

def poly_interpolation_2d(p,a,b,c,d,X,Y,n,m,f,produce_fig):
    """
    Finds the 2D polynomial interpolant of order p of a function f
    using uniformly spaced points over the rectangle [a,b]x[c,d]
    
    Optionally draws a contour plot of this interpolant

    Parameters
    ----------
    a, b, c, d : floats
          limits on which the interpolant is based (a < b and c < d assumed)
    p : int
          polynomial degree to use (assumed positive)
    n, m : int
          number of points at which to evaluate the interpolant in y and x
          directions, respectively
    X, Y : numpy.ndarray of shape (m,n)
          points at which to evaluate the interpolant
    f : function of one variable
          function to find the interpolant of
    produce_fig : bool
          True - plot the interpolant, False - don't

    Returns
    -------
    interpolant : numpy.ndarray of shape (m,n)
        The interpolant evaluated at all points in in x
    fig :: matplotlib.matplotlib.figure.Figure/None
        Plot of interpolant
    """

    #Compute Lagrange polys in x direction
    xhat = np.linspace(a,b,p+1)
    Lx, error_flag = lp.lagrange_poly(p,xhat,n,X,1.0e-10)

    #Compute Lagrange polys in y direction
    yhat = np.linspace(c,d,p+1)
    Ly, error_flag = lp.lagrange_poly(p,yhat,n,Y,1.0e-10)

    #Construct 2d Lagrange functions at points of X,Y
    interpolant = np.zeros(X.shape)

    for j, xhat_j in enumerate(xhat):
        for k, yhat_k in enumerate(yhat):
            interpolant += f(xhat_j,yhat_k)*Lx[j]*Ly[k]

    if produce_fig:
        fig = plt.figure()
        plt.contour(X,Y,interpolant,20)
        plt.xlim([a,b])
        plt.ylim([c,d])
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("2D Interpolant, degree p = "+str(p))
        plt.axis("scaled")
    else:
        fig = None

    return interpolant,fig  

####################################################################

def approximate_derivative(x,p,h,k,f):
    """
    Finds the aproximation to the derivative f'(x) based on
    differentiating the pth order polynomial interpolant

    Parameters
    ----------
    x : float
          point at which to approximate the derivative
    p : int
          polynomial degree to use (assumed positive)
    h : float
          mesh spacing to use
    k : int
          which interpolant to use. 0 <= k <= p. 
    f : function
        function to approximate

    Returns
    -------
    deriv_approx : float
        approximation of the derivative
    """
            
    #Construct the nodal points based on k and p
    nodes = np.linspace(x-k*h,x+(p-k)*h,p+1)

    #Evaulate the funtion at the nodes
    y = f(nodes)

    #Construct the differentiated Lagrange polynomials
    deriv_lagrange = lp.deriv_lagrange_poly(p,nodes,1,x,1.0e-10)[0]

    #Evaluate the approximate derivative
    deriv_approx = np.matmul(y,deriv_lagrange)[0]

    return deriv_approx


####################################################################

print("\nAny outputs above this line are due to importing lagrange_polynomials.py.\n")

################
## Test Code ##
################

################
#%% Q2 Test
################

# Initialise
a = 0.5
b = 1.5
p = 3
n = 10
x = np.linspace(0.5,1.5,n)
f = lambda x: np.exp(x)+np.sin(np.pi*x)
#Run the function
interpolant, fig = poly_interpolation(a,b,p,n,x,f,False)

print("\n################################")
print('Q2 TEST OUTPUT:\n')
print("interpolant = \n")
print(interpolant)

################
#%% Q4 Test
################

f = lambda x,y : np.exp(x**2+y**2)
n = 4; m = 3
a = 0; b = 1; c = -1; d = 1 
x = np.linspace(a,b,n)
y = np.linspace(c,d,m)
X,Y = np.meshgrid(x,y)

interpolant,fig = poly_interpolation_2d(11,a,b,c,d,X,Y,n,m,f,False)

print("\n################################")
print('Q4 TEST OUTPUT:\n')
print("interpolant = \n")
print(interpolant)

################
#%% Q6 Test
################

print("\n################################")
print("Q6 TEST OUTPUT:\n")
#Initialise
p = 3
h = 0.1
x = 0.5
f = lambda x: np.cos(np.pi*x)+x

for k in range(4):
    #Run test 
    deriv_approx = approximate_derivative(x,p,h,k,f)
    print("k = " + str(k)+ ", deriv_approx = " + str(deriv_approx))