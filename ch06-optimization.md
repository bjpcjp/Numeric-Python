# Optimization
* How to find optimal elements from set of candidates
* Usually stated as finding min or max of function in given domain
- __Problem Classification__
- __Univariate Optimization__
- __Multivariate Optimization (Unconstrained)__
- __Nonlinear Least Square Problems__
- __Constrained Optimization__
- __Linear Programming__


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
import numpy as np
import sympy
from scipy import optimize     # nonlinear optimization
import cvxopt                  # convex optimization
```

[cvxopt library](http://stanford.edu/~boyd/cvxbook)


```python
sympy.init_printing()
```


```python
from __future__ import division
```

### Problem Classification
* Description:
    * minimization of f(x)
    * subject to m sets of equality constraints g(x)=0
    * subject to p sets of inequality constraints h(x)<=0
* General formulation == no good solver methods
* Some methods available for special cases
* *linear programming* problem: if f(x) & constraints are linear.
* *convex* nonlinear problems: only one global minimum
![pic](pics/convex-vs-nonconvex.png)

### Univariate optimization
* Bracketing & Newton's method can be applied
* __golden section search__: used in SciPy.optimize __golden()__. Relatively safe, but slow convergence.
* SciPy.optimize __brent()__: a hybrid of golden & Newton's
* General-purpose: __minimize_scalar()__ with method="golden","brent" or "bounded"


```python
# example problem: minimize area of cylinder with unit volume
# r = radius, h = height, f(r,h) = 2*pi*r^2 + 2*pi*r*h
# 2D optimization problem with an equality constraint

r, h = sympy.symbols("r, h")
```


```python
Area   = 2*sympy.pi*r**2 + 2*sympy.pi*r*h
Volume =                     sympy.pi*r**2*h
```


```python
h_r = sympy.solve(Volume - 1)[0]
```


```python
Area_r = Area.subs(h_r)
```


```python
rsol = sympy.solve(Area_r.diff(r))[0]
rsol
```




$\displaystyle \frac{2^{\frac{2}{3}}}{2 \sqrt[3]{\pi}}$




```python
_.evalf()
```




$\displaystyle 0.541926070139289$



* Now verify 2nd derivative is positive and that __rsol__ corresponds to a minimum.


```python
Area_r.diff(r, 2).subs(r, rsol)
```




$\displaystyle 12 \pi$




```python
Area_r.subs(r, rsol)
```




$\displaystyle 3 \cdot \sqrt[3]{2} \sqrt[3]{\pi}$




```python
_.evalf()
```




$\displaystyle 5.53581044593209$



### Solve numerically
* Typically required for more realistic problems.


```python
# define an objective function
def f(r):
    return 2 * np.pi * r**2 + 2 / r
```


```python
# solve using optimize.brent()
r_min = optimize.brent(f, brack=(0.1, 4))
r_min, f(r_min)
```




$\displaystyle \left( 0.541926077255714, \  5.53581044593209\right)$



* Instead of calling __optimize.brent()__ directly, can use generic interface __optimize.minimize_scalar()__. Use __bracket__ to specify a starting interval.


```python
# radius that minimizes cylinder area ~ 0.54;
# corresponding min area ~ 5.54
optimize.minimize_scalar(f, bracket=(0.1, 5))
```




     message: 
              Optimization terminated successfully;
              The returned value satisfies the termination criteria
              (using xtol = 1.48e-08 )
     success: True
         fun: 5.535810445932086
           x: 0.5419260648976671
         nit: 13
        nfev: 16



* Graph optimization across a range of r:


```python
r = np.linspace(0, 2, 100)
```


```python
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(r, f(r), lw=2, color='b')
ax.plot(r_min, f(r_min), 'r*', markersize=15)
ax.set_title(r"$f(r) = 2\pi r^2+2/r$", fontsize=18)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_ylim(0, 30)

fig.tight_layout()
fig.savefig('ch6-univariate-optimization-example.pdf')
```

    /tmp/ipykernel_64789/914950341.py:3: RuntimeWarning: divide by zero encountered in divide
      return 2 * np.pi * r**2 + 2 / r



    
![png](ch06-optimization_files/ch06-optimization_25_1.png)
    


### Unconstrained Multivariate Optimization
* Much harder. Analytical & bracketing schemes are rarely feasible.
* Instead consider using __gradient descent__ method.
* Guaranteed to converge on a minimum, but show & prone to overshoot (ie, "zigzag").
* Newton's method can help. Can be viewed as local approximation of the function. Requires finding both the gradient & Hessian of the function.
* SciPy: Newton's implemented with __optimize.fmin_ncg()__. 


```python
x1, x2 = sympy.symbols("x_1, x_2")
```


```python
# objective function
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2
```


```python
# gradient
fprime_sym = [f_sym.diff(x_) 
              for x_ in (x1, x2)]

sympy.Matrix(fprime_sym)
```




$\displaystyle \left[\begin{matrix}- 2 x_{2} + 4 \left(x_{1} - 1\right)^{3}\\- 2 x_{1} + 10 x_{2} - 10\end{matrix}\right]$




```python
# hessian
fhess_sym = [
    [f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] 
    for x2_ in (x1, x2)]

sympy.Matrix(fhess_sym)
```




$\displaystyle \left[\begin{matrix}12 \left(x_{1} - 1\right)^{2} & -2\\-2 & 10\end{matrix}\right]$




```python
# use symbolic expressions to create vectorized functions for them
f_lmbda      = sympy.lambdify((x1, x2), f_sym, 'numpy')
fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')
fhess_lmbda  = sympy.lambdify((x1, x2), fhess_sym, 'numpy')
```


```python
# funcs returned by sympy.lambdify take one arg for each var
# SciPy optimization func expect a vectorized function.
# need a wrapper.

def func_XY_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array(f(X[0], X[1]))
```


```python
f      = func_XY_X_Y(f_lmbda)
fprime = func_XY_X_Y(fprime_lmbda)
fhess  = func_XY_X_Y(fhess_lmbda)
```


```python
# optimize using (0,0) as a starting point
X_opt = optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)
```

    Optimization terminated successfully.
             Current function value: -3.867223
             Iterations: 8
             Function evaluations: 10
             Gradient evaluations: 10
             Hessian evaluations: 8



```python
X_opt # minimum x1,x2
```




    array([1.88292613, 1.37658523])



* Below: visualize the objective function and solution, using a contour plot.


```python
fig, ax = plt.subplots(figsize=(8, 6))
x_ = y_ = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, f_lmbda(X, Y), 50)
ax.plot(X_opt[0], X_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()
fig.savefig('ch6-examaple-two-dim.pdf');
```


    
![png](ch06-optimization_files/ch06-optimization_37_0.png)
    


* Not always possible to provide functions for gradients & Hessians. Sometimes better to estimate them.
* __BFGS (optimize.fmin_bfgs())__ and __conjugate-gradient (optimize.fmin_cg())__ methods help here.
* Rule of thumb: BFGS still a good starting point

### Brute force search for initial point
* Suitable when problem space has *many* local minima


```python
# objective function
def f(X):
    x, y = X
    return (4*np.sin(np.pi*x) + 6*np.sin(np.pi*y)) + (x-1)**2 + (y-1)**2
```


```python
# brute-force search:
# slice objects == coordinate grid search space
# finish=None == auto-refine best candidate

x_start = optimize.brute(f, 
                         (slice(-3, 5, 0.5), 
                          slice(-3, 5, 0.5)), 
                         finish=None)
x_start, f(x_start)
```




    (array([1.5, 1.5]), -9.5)




```python
# we now have good starting point for interative solver like BFGS
x_opt = optimize.fmin_bfgs(f, x_start)
```

    Optimization terminated successfully.
             Current function value: -9.520229
             Iterations: 4
             Function evaluations: 21
             Gradient evaluations: 7



```python
x_opt, f(x_opt)
```




    (array([1.47586906, 1.48365787]), -9.520229273055016)




```python
# visualize solution
# need wrapper to shuffle params

def func_X_Y_to_XY(f, X, Y):
    s = np.shape(X)
    return f(
        np.vstack(
            [X.ravel(), Y.ravel()])).reshape(*s)
```


```python
fig, ax = plt.subplots(figsize=(8,6))
x_ = y_ = np.linspace(-3, 5, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 25)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()
fig.savefig('ch6-example-2d-many-minima.pdf');
```


    
![png](ch06-optimization_files/ch06-optimization_45_0.png)
    


### Nonlinear least square problems
* Most popular solver method: __Levenberg-Marquardt__
* SciPy's __optimize.leastsq()__ uses Levenberg-Marquardt.


```python
def f(x, beta0, beta1, beta2):
    return beta0 + beta1 * np.exp(-beta2 * x**2)
```


```python
beta = (0.25, 0.75, 0.5)
```


```python
# generate random datapoints
xdata = np.linspace(0, 5, 50)
y = f(xdata, *beta)
ydata = y + 0.05 * np.random.randn(len(xdata))
```


```python
# start solver by defining function for residuals
def g(beta):
    return ydata - f(xdata, *beta)
```


```python
# define initial guess for parameter vector
beta_start = (1, 1, 1)
# let leastsq() solve it
beta_opt, beta_cov = optimize.leastsq(g, beta_start)
```


```python
# results
beta_opt
```




    array([0.23782674, 0.78424045, 0.48718179])




```python
fig, ax = plt.subplots()

ax.scatter(xdata, ydata)
ax.plot(xdata, y, 'r', lw=2)
ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2)
ax.set_xlim(0, 5)
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)

fig.tight_layout()
fig.savefig('ch6-nonlinear-least-square.pdf')
```


    
![png](ch06-optimization_files/ch06-optimization_53_0.png)
    


* alternative method: __curve_fit()__ - convenience wrapper around __leastsq()__; eliminates need to explicitly define residual function for the least square problem.


```python
beta_opt, beta_cov = optimize.curve_fit(f, xdata, ydata)
beta_opt
```




    array([0.23782674, 0.78424045, 0.48718179])



## Constrained optimization

* Simple example: optimization with coordinates subject to boundary conditions
* SciPy offers __L-BFGS-B__ method for this use case


```python
# objective function
def f(X):
    x, y = X
    return (x-1)**2 + (y-1)**2
```


```python
x_opt = optimize.minimize(
    f, (0, 0), 
    method='BFGS').x
```


```python
# boundary constraints
bnd_x1, bnd_x2 = (2, 3), (0, 2)

x_cons_opt = optimize.minimize(
    f, np.array([0, 0]), 
    method='L-BFGS-B', 
    bounds=[bnd_x1, bnd_x2]).x
```

* Visualization of the objective function shown below.
* unconstrained minima = blue star
* constrainted minima = red star
* feasible constrained region = gray shading


```python
fig, ax = plt.subplots(figsize=(8,6))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]), 
                           bnd_x1[1] - bnd_x1[0], bnd_x2[1] - bnd_x2[0],
                           facecolor="grey")
ax.add_patch(bound_rect)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)

fig.tight_layout()
fig.savefig('ch6-example-constraint-bound.pdf');
```


    
![png](ch06-optimization_files/ch06-optimization_62_0.png)
    


### Lagrange multipliers
* A technique for converting constrained optimization into unconstrained equivalent by intro'ing more variables.
* Below: maximize volume of rectange of dimensions x1,x2,x3 with constraint that total surface area must be 1.0.


```python
# symbolics first
x = x1, x2, x3, l = sympy.symbols("x_1, x_2, x_3, lambda")
```


```python
# volume
f = x1*x2*x3
```


```python
# surface area constraint
g = 2 * (x1*x2 + x2*x3 + x3*x1) - 1
```


```python
# Lagrangian
L = f + l*g
```


```python
# Lagrangian gradient
grad_L = [sympy.diff(L, x_) 
          for x_ in x]
```


```python
# solve for zero. should return two points.
# However, 2nd point has x1<0 = not viable use case. (x1 is a dimension)
# so 1st point must be answer.
sols = sympy.solve(grad_L)
sols
```




$\displaystyle \left[ \left\{ \lambda : - \frac{\sqrt{6}}{24}, \  x_{1} : \frac{\sqrt{6}}{6}, \  x_{2} : \frac{\sqrt{6}}{6}, \  x_{3} : \frac{\sqrt{6}}{6}\right\}, \  \left\{ \lambda : \frac{\sqrt{6}}{24}, \  x_{1} : - \frac{\sqrt{6}}{6}, \  x_{2} : - \frac{\sqrt{6}}{6}, \  x_{3} : - \frac{\sqrt{6}}{6}\right\}\right]$




```python
# verify by eval'ing constraint func & objective func using answer
g.subs(sols[0]), f.subs(sols[0])
```




$\displaystyle \left( 0, \  \frac{\sqrt{6}}{36}\right)$



### Inequality constraint solver
* SciPy offers __optimize.slsqp() (sequential least squares programming)__, also available via __optimize.minimize__ with __method='SLSQP'__.


```python
# objective function
def f(X):
    return -X[0] * X[1] * X[2]

# constraint function
def g(X):
    return 2 * (X[0]*X[1] + X[1] * X[2] + X[2] * X[0]) - 1
```


```python
constraints = [dict(type='eq', fun=g)] # type = 'eq'
```


```python
result = optimize.minimize(
    f, [0.5, 1, 1.5], 
    method='SLSQP', 
    constraints=constraints)
result
```




     message: Optimization terminated successfully
     success: True
      status: 0
         fun: -0.06804136862287297
           x: [ 4.082e-01  4.083e-01  4.083e-01]
         nit: 18
         jac: [-1.667e-01 -1.667e-01 -1.667e-01]
        nfev: 77
        njev: 18




```python
result.x
```




    array([0.40824188, 0.40825127, 0.40825165])



### Inequality constraints
* Done using __type='ineq'__ in a constraint dictionary.
* Below: quadratic problem with inequality constraint. 


```python
# objective function
def f(X):
    return (X[0] - 1)**2 + (X[1] - 1)**2
# constraint function
def g(X):
    return X[1] - 1.75 - (X[0] - 0.75)**4
```


```python
x_opt = optimize.minimize(
    f, (0, 0), method='BFGS').x
```


```python
constraints = [dict(type='ineq', fun=g)] # type = 'ineq'
```


```python
x_cons_opt = optimize.minimize(
    f, (0, 0), 
    method='SLSQP', 
    constraints=constraints).x
```


```python
fig, ax = plt.subplots(figsize=(8,6))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)

ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)
ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color="grey")
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)

ax.set_ylim(-1, 3)
ax.set_xlabel(r"$x_0$", fontsize=18)
ax.set_ylabel(r"$x_1$", fontsize=18)
plt.colorbar(c, ax=ax)

fig.tight_layout()
fig.savefig('ch6-example-constraint-inequality.pdf');
```


    
![png](ch06-optimization_files/ch06-optimization_81_0.png)
    


* Alternative solver: for optimization problems with __only__ inequality constraints.
* Constrained optimization by linear approximation (COBYLA). Implemented by replacing __method='SLSQP'__ with __method='COBYLA'__.


```python
x_cons_opt = optimize.minimize(
    f, (0, 0), 
    method='COBYLA', 
    constraints=constraints).x
```

### Linear programming
* Much more restricted type of optimization problem.
* Linear objective function
* All constraints are linear equalities/inequalities.
* Can be solved much more efficiently than general nonlinear problems.
* __Simplex__: a popular solver method.
* Let's use the __cvxopt__ library's __solvers.lp__ function.


```python
c = np.array([-1.0, 2.0, -3.0])

A = np.array([[ 1.0, 1.0, 0.0],
              [-1.0, 3.0, 0.0],
              [ 0.0, -1.0, 1.0]])

b = np.array([1.0, 2.0, 3.0])
```


```python
# cvxopt has unique classes for matrices & vectors - can talk to NumPy
A_ = cvxopt.matrix(A)
b_ = cvxopt.matrix(b)
c_ = cvxopt.matrix(c)
```


```python
sol = cvxopt.solvers.lp(c_, A_, b_); sol
```

    Optimal solution found.





    {'x': <3x1 matrix, tc='d'>,
     'y': <0x1 matrix, tc='d'>,
     's': <3x1 matrix, tc='d'>,
     'z': <3x1 matrix, tc='d'>,
     'status': 'optimal',
     'gap': 0.0,
     'relative gap': 0.0,
     'primal objective': -10.0,
     'dual objective': -10.0,
     'primal infeasibility': 0.0,
     'primal slack': -0.0,
     'dual slack': 0.0,
     'dual infeasibility': 1.4835979218054372e-16,
     'residual as primal infeasibility certificate': None,
     'residual as dual infeasibility certificate': None,
     'iterations': 0}




```python
x = np.array(sol['x']); x
```




    array([[0.25],
           [0.75],
           [3.75]])




```python
sol['primal objective']
```




$\displaystyle -10.0$


