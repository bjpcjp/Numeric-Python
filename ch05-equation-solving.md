# Equation solving
## Contents
- __Linear Equation Systems__
- __Square Systems__
- __Rectangular Systems__
- __Eigenvalue Problems__
- __Nonlinear Equations__
- __Univariate Equations__
- __Nonlinear Equation Systems__


```python
from scipy import linalg as la
from scipy import optimize
import sympy
import numpy as np
```


```python
sympy.init_printing()
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = "12"
```


```python
from __future__ import division
```

### Linear Algebra - Linear Equation Systems
* Linear Algebra modules exist in SymPy, NumPy (numpy.linalg) and SciPy (scipy.linalg).
* Typical notation: A (m x n) * x (n x 1) = b (m x 1).
* A system with **n<m** is *underdetermined* (less equations than unknowns, therefore can't find unique solution)

### Square Systems (m = n)
* Indicates situation with #equations == #unknowns; potentially has unique solution.
* For unique solution, A must be *non-singular* (A_inverse exists). 
* If A is *singular* (__rank(A)__ < n), or if __det(A)__ == 0, then Ax=b can have either no solution or infinitely many solutions. Finding rank(A) is therefore useful.
* When A has full rank, solution is guaranteed to exist but NOT guaranteed to be computable. __cond(A)__ is a good indicator of solution accuracy.

* Example: system of two linear equations

$$
2 x_1 + 3 x_2 = 4
$$

$$
5 x_1 + 4 x_2 = 3
$$


```python
fig, ax = plt.subplots(figsize=(8, 4))

x1 = np.linspace(-4, 2, 100)

x2_1 = (4 - 2 * x1)/3
x2_2 = (3 - 5 * x1)/4

ax.plot(x1, x2_1, 'r', lw=2, label=r"$2x_1+3x_2-4=0$")
ax.plot(x1, x2_2, 'b', lw=2, label=r"$5x_1+4x_2-3=0$")

A = np.array([[2, 3], [5, 4]])
b = np.array([4, 3])
x = la.solve(A, b)

ax.plot(x[0], x[1], 'ko', lw=2)
ax.annotate("The intersection point of\nthe two lines is the solution\nto the equation system",
            xy=(x[0], x[1]), xycoords='data',
            xytext=(-120, -75), textcoords='offset points', 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.3"))

ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
ax.legend();

fig.tight_layout()
fig.savefig('ch5-linear-systems-simple.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_8_0.png)
    


* Direct approach: find inv(A), multiply by b. (Not the most efficient approach.)
* Better: **LU factorization of A** (A=LU; L = lower triangular matrix, U = upper triangular matrix. First solve Ly=b, then solve Ux=y.)
* Symbolic & numerical approaches shown below.
* **TODO**: link here to LU definition.

### Symbolic approach


```python
A = sympy.Matrix([[2, 3], [5, 4]])
b = sympy.Matrix([4, 3])
```


```python
A.rank(), A.condition_number()
```




$\displaystyle \left( 2, \  \frac{\sqrt{2 \sqrt{170} + 27}}{\sqrt{27 - 2 \sqrt{170}}}\right)$




```python
A.condition_number()
```




$\displaystyle \frac{\sqrt{2 \sqrt{170} + 27}}{\sqrt{27 - 2 \sqrt{170}}}$




```python
sympy.N(_)
```




$\displaystyle 7.58240137440151$




```python
A.norm()
```




$\displaystyle 3 \sqrt{6}$



### Better method: LU Factorization of A, such that A = LU
- L = lower triangular matrix
- U = upper triangular matrix


```python
L, U, _ = A.LUdecomposition()
```


```python
L, U, L*U
```




$\displaystyle \left( \left[\begin{matrix}1 & 0\\\frac{5}{2} & 1\end{matrix}\right], \  \left[\begin{matrix}2 & 3\\0 & - \frac{7}{2}\end{matrix}\right], \  \left[\begin{matrix}2 & 3\\5 & 4\end{matrix}\right]\right)$




```python
x = A.solve(b)
x
```




$\displaystyle \left[\begin{matrix}-1\\2\end{matrix}\right]$



### Numerical approach


```python
A = np.array([[2, 3], [5, 4]])
b = np.array([4, 3])
```


```python
np.linalg.matrix_rank(A)
```




    2




```python
np.linalg.cond(A), np.linalg.norm(A)
```




$\displaystyle \left( 7.58240137440152, \  7.34846922834953\right)$



* SciPy linear algebra's __la.lu__ function - returns permutation matrix P and L,U matrices such that __A = PLU__.
* Can solve linear __Ax = b__ systems using __la.solve__. This is a preferred method.


```python
P, L, U = la.lu(A)
```


```python
L
```




    array([[1. , 0. ],
           [0.4, 1. ]])




```python
U
```




    array([[5. , 4. ],
           [0. , 1.4]])




```python
L*U
```




    array([[5. , 0. ],
           [0. , 1.4]])




```python
la.solve(A, b)
```




    array([-1.,  2.])



### Example : rank and condition numbers -> numerical errors
* SymPy: can obtain exact results, but not all problems are symbolically solvable.
* NumPy/SciPy: guaranteed to get a result. Probably approximate due to FP math errors.
* Below: illustrates differences between symbolic & numerical approaches


```python
p = sympy.symbols("p", positive=True)
```


```python
A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]])
```


```python
b = sympy.Matrix([1, 2])
```


```python
sympy.simplify(A.solve(b))
```




$\displaystyle \left[\begin{matrix}\frac{2 p - 1}{p - 1}\\- \frac{\sqrt{p}}{p - 1}\end{matrix}\right]$




```python
# Symbolic problem specification
p = sympy.symbols("p", positive=True)
A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]])
b = sympy.Matrix([1, 2])

# Solve symbolically
x_sym_sol = A.solve(b)
x_sym_sol.simplify()
x_sym_sol
Acond = A.condition_number().simplify()

# Function for solving numerically
AA = lambda p: np.array([[1, np.sqrt(p)], [1, 1/np.sqrt(p)]])
bb = np.array([1, 2])
x_num_sol = lambda p: np.linalg.solve(AA(p), bb)

# Graph the difference between the symbolic (exact) and numerical results.
p_vec = np.linspace(0.9, 1.1, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for n in range(2):
    x_sym = np.array([x_sym_sol[n].subs(p, pp).evalf() for pp in p_vec])
    x_num = np.array([x_num_sol(pp)[n] for pp in p_vec])
    axes[0].plot(p_vec, (x_num - x_sym)/x_sym, 'k')
axes[0].set_title("Error in solution\n(numerical - symbolic)/symbolic")
axes[0].set_xlabel(r'$p$', fontsize=18)

axes[1].plot(p_vec, [Acond.subs(p, pp).evalf() for pp in p_vec])
axes[1].set_title("Condition number")
axes[1].set_xlabel(r'$p$', fontsize=18)

fig.tight_layout()
fig.savefig('ch5-linear-systems-condition-number.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_35_0.png)
    


### Rectangular systems:
* #vars > #equations, so solution must be given in terms of remaining vars = symbolic approach.
* Example: 3 unknowns, two equations:
![pic](pics/rectangular-linear-equation.png)


```python
unknown = sympy.symbols("x, y, z")
A =       sympy.Matrix([[1, 2, 3], [4, 5, 6]])
x =       sympy.Matrix(unknown)
b =       sympy.Matrix([7, 8])
```


```python
AA = A * x - b
```


```python
# solution: defined line in 3D space.
# Any point on line satisfies equations.
sympy.solve(A*x - b, unknown)
```




$\displaystyle \left\{ x : z - \frac{19}{3}, \  y : \frac{20}{3} - 2 z\right\}$



### Overdetermined systems: least squares
* #equations > #vars: no exact solution. Approximate solution (ie, "data fitting") *could* be interesting.
* Can use **least squares** to find curve with min sum of squared error. SymPy: __solve_least_squares()__; SciPy: __la.lstsq()__.


```python
# example:
# y = quadratic polynomial, y = A + Bx + Cx^2
# goal: fit model to experimental data

np.random.seed(1234)

# define true model parameters
x = np.linspace(-1, 1, 100)
a, b, c = 1, 2, 3
y_exact = a + b * x + c * x**2

# simulate noisy data points
m = 100
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + np.random.randn(m)

# fit the data to the model using linear least square
A = np.vstack([X**0, X**1, X**2])  # see np.vander for alternative
sol, r, rank, sv = la.lstsq(A.T, Y)
y_fit = sol[0] + sol[1] * x + sol[2] * x**2
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2);

fig.savefig('ch5-linear-systems-least-square.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_41_0.png)
    


### Model fit vs underlying process
* Previous data compared to linear & 15th-order polynomial model
* Linear: underfit (too simple)
* Polynomial: overfit (fits model, but also fits noise)
* Lesson learned: **pick the right model**.


```python
# fit the data to the model using linear least square: 

# 1st order polynomial
A = np.vstack([X**n for n in range(2)])
sol, r, rank, sv = la.lstsq(A.T, Y)
y_fit1 = sum([s * x**n for n, s in enumerate(sol)])

# 15th order polynomial
A = np.vstack([X**n for n in range(16)])
sol, r, rank, sv = la.lstsq(A.T, Y)
y_fit15 = sum([s * x**n for n, s in enumerate(sol)])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
ax.plot(x, y_fit1, 'b', lw=2, label='Least square fit [1st order]')
ax.plot(x, y_fit15, 'm', lw=2, label='Least square fit [15th order]')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2);

fig.savefig('ch5-linear-systems-least-square-2.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_43_0.png)
    


### Eigenvalue problems
* Ax = (phi)x
* A = (N x N)
* x = unknown (vector = eigenvector)
* phi = unknown (scalar = eigenvalue of A)
* In SymPy: __eigenvals()__ and __eigenvects()__


```python
eps, delta = sympy.symbols("epsilon, delta")
```


```python
H = sympy.Matrix([[eps, delta], [delta, -eps]])
H
```




$\displaystyle \left[\begin{matrix}\epsilon & \delta\\\delta & - \epsilon\end{matrix}\right]$




```python
# eigenvals() -- returns dict
# keys = eigenvalues, vals = multiplicity of eigenvalue

eval1, eval2 = H.eigenvals()
eval1, eval2
```




$\displaystyle \left( - \sqrt{\delta^{2} + \epsilon^{2}}, \  \sqrt{\delta^{2} + \epsilon^{2}}\right)$




```python
# eigenvects() -- returns list of tuples
# each tuple -- eigenvalue, multiplicity of eigenvalue, list of eigenvectors
H.eigenvects()
```




$\displaystyle \left[ \left( - \sqrt{\delta^{2} + \epsilon^{2}}, \  1, \  \left[ \left[\begin{matrix}\frac{\epsilon}{\delta} - \frac{\sqrt{\delta^{2} + \epsilon^{2}}}{\delta}\\1\end{matrix}\right]\right]\right), \  \left( \sqrt{\delta^{2} + \epsilon^{2}}, \  1, \  \left[ \left[\begin{matrix}\frac{\epsilon}{\delta} + \frac{\sqrt{\delta^{2} + \epsilon^{2}}}{\delta}\\1\end{matrix}\right]\right]\right)\right]$




```python
(eval1, _, evec1), (eval2, _, evec2) = H.eigenvects()
```


```python
sympy.simplify(evec1[0].T * evec2[0])
```




$\displaystyle \left[\begin{matrix}0\end{matrix}\right]$



* Symbolic evaluation == nice, but only works for small matrices. 
* Anything > 3x3 very cumbersome even in SymPy. Go with numerical approach instead. (SciPy __la.eigvals__, __la.eig__.)
* Below: solve a numerical eigenvalue problem with __la.eig__.


```python
A = np.array([[1, 3, 5], [3, 5, 3], [5, 3, 9]])
A
```




    array([[1, 3, 5],
           [3, 5, 3],
           [5, 3, 9]])




```python
evals, evecs = la.eig(A)
```


```python
evals
```




    array([13.35310908+0.j, -1.75902942+0.j,  3.40592034+0.j])




```python
evecs
```




    array([[ 0.42663918,  0.90353276, -0.04009445],
           [ 0.43751227, -0.24498225, -0.8651975 ],
           [ 0.79155671, -0.35158534,  0.49982569]])




```python
la.eigvalsh(A)
```




    array([-1.75902942,  3.40592034, 13.35310908])



### Nonlinear equations
* Much broader than linear functions.
* Can always be written as f(x)=0 -- solvers often called "root finders".
* In general, not analytically solvable. Consider numerical sol'ns first. Graphing often provides valuable clues.

### Univariate (single dependent variable, "x" = scalar)


```python
x = np.linspace(-2, 2, 1000)

#
# plot four examples of nonlinear functions
#

f1 = x**2 - x - 1                        # two roots
f2 = x**3 - 3 * np.sin(x)                # three roots
f3 = np.exp(x) - 2                       # one root
f4 = 1 - x**2 + np.sin(50 / (1 + x**2))  # "many" roots

fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

for n, f in enumerate([f1, f2, f3, f4]):
    axes[n].plot(x, f, lw=1.5)
    axes[n].axhline(0, ls=':', color='k')
    axes[n].set_ylim(-5, 5)
    axes[n].set_xticks([-2, -1, 0, 1, 2])
    axes[n].set_xlabel(r'$x$', fontsize=18)

axes[0].set_ylabel(r'$f(x)$', fontsize=18)

titles = [r'$f(x)=x^2-x-1$', 
          r'$f(x)=x^3-3\sin(x)$',
          r'$f(x)=\exp(x)-2$', 
          r'$f(x)=\sin\left(50/(1+x^2)\right)+1-x^2$']

for n, title in enumerate(titles):
    axes[n].set_title(title)
    
fig.tight_layout()
fig.savefig('ch5-nonlinear-plot-equations.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_59_0.png)
    


### Root solver techniques
* Iteration-based - eval function until solution has desired accuracy.
* Two standard methods: __bisection__, __Newton__.

### Bisection method
* start interval [a,b]; f(a), f(b) == different signs (guarantees at least one root)


```python
# define a function, desired tolerance and starting interval [a, b]
f = lambda x: np.exp(x) - 2
tol = 0.1
a, b = -2, 2
x = np.linspace(-2.1, 2.1, 1000)

# graph the function f
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax.plot(x, f(x), lw=1.5)
ax.axhline(0, ls=':', color='k')
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$f(x)$', fontsize=18)

# find root using bisection method and visualize steps

fa, fb = f(a), f(b)

ax.plot(a, fa, 'ko')
ax.plot(b, fb, 'ko')
ax.text(a, fa + 0.5, r"$a$", ha='center', fontsize=18)
ax.text(b, fb + 0.5, r"$b$", ha='center', fontsize=18)

n = 1
while b - a > tol:
    m = a + (b - a)/2
    fm = f(m)

    ax.plot(m, fm, 'ko')
    ax.text(m, fm - 0.5, r"$m_%d$" % n, ha='center')
    n += 1
    
    if np.sign(fa) == np.sign(fm):
        a, fa = m, fm
    else:
        b, fb = m, fm

ax.plot(m, fm, 'r*', markersize=10)
ax.annotate("Root approximately at %.3f" % m,
            fontsize=14, family="serif",
            xy=(a, fm), xycoords='data',
            xytext=(-150, +50), textcoords='offset points', 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.5"))

ax.set_title("Bisection method")

fig.tight_layout()
fig.savefig('ch5-nonlinear-bisection.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_62_0.png)
    


### Newton's method
* Converges faster than bisection method
* Approximates f(x) with 1st order Taylor expansion (a linear function)


```python
# define a function, desired tolerance and starting point xk

tol = 0.01
xk = 2

s_x = sympy.symbols("x")
s_f = sympy.exp(s_x) - 2

# using SymPy to symbolically find derivatives
# not always possible - all-numerical alternative may be needed.

f =  lambda x: sympy.lambdify(s_x, s_f, 'numpy')(x)
fp = lambda x: sympy.lambdify(s_x, sympy.diff(s_f, s_x), 'numpy')(x)

x = np.linspace(-1, 2.1, 1000)

# setup a graph for visualizing the root finding steps

fig, ax = plt.subplots(1, 1, figsize=(12,4))

ax.plot(x, f(x))
ax.axhline(0, ls=':', color='k')

# repeat Newton's method 
# until convergence to desired tolerance reached

n = 0
while f(xk) > tol:
    xk_new = xk - f(xk) / fp(xk)

    ax.plot([xk, xk], [0, f(xk)], color='k', ls=':')
    ax.plot(xk, f(xk), 'ko')
    ax.text(xk, -.5, r'$x_%d$' % n, ha='center')
    ax.plot([xk, xk_new], [f(xk), 0], 'k-')

    xk = xk_new
    n += 1

ax.plot(xk, f(xk), 'r*', markersize=15)
ax.annotate("Root approximately at %.3f" % xk,
            fontsize=14, family="serif",
            xy=(xk, f(xk)), xycoords='data',
            xytext=(-150, +50), textcoords='offset points', 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.5"))

ax.set_title("Newton's method")
ax.set_xticks([-1, 0, 1, 2])
fig.tight_layout()
fig.savefig('ch5-nonlinear-newton.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_64_0.png)
    


### Newton's method - limitation
* Requires function values & function derivatives at each iteration.
* When using SymPy, no problem.
* When all-numerical implementation - not possible. Needs numerical derivatives instead.
* __Secant__ method: uses two previous function evals to find a linear approximation of the function, which can be used to find a new estimate of the root.


```python
# bisect method
# args: function, interval-low, interval-high
optimize.bisect(lambda x: np.exp(x) - 2, -2, 2)
```




$\displaystyle 0.693147180560118$




```python
# newton's method
# args: function, initial root guess
optimize.newton(lambda x: np.exp(x) - 2, 2)
```




$\displaystyle 0.693147180559946$




```python
# newton's method - optional arg for function derivative
# if fprime provided, newton's method used
# if fprime not used, secant method used

x_root_guess = 2
f            = lambda x: np.exp(x) - 2
fprime       = lambda x: np.exp(x)
```


```python
optimize.newton(f, x_root_guess)
```




$\displaystyle 0.693147180559946$




```python
optimize.newton(f, x_root_guess, fprime=fprime)
```




$\displaystyle 0.693147180559945$




```python
# brentq method: variants of bisect method
# preferred all-around root finder in SciPy

optimize.brentq(lambda x: np.exp(x) - 2, -2, 2)
```




$\displaystyle 0.693147180559945$




```python
# brenth method: another bisect method variant
optimize.brenth(lambda x: np.exp(x) - 2, -2, 2)
```




$\displaystyle 0.693147180559938$




```python
# ridder method
optimize.ridder(lambda x: np.exp(x) - 2, -2, 2)
```




$\displaystyle 0.69314718055904$



### Multivariate Nonlinear Equations
* Much more complicated to solve. No method that guarantees convergence.
* Not all univariate methods can be generalized to multivariate problems.


```python
# ex: system of 2 multivariate, nonlinear equations
def f(x):
    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]
```


```python
# fsolve(function, initial_guess)
optimize.fsolve(f, [1, 1])
```




    array([0.73205081, 0.46410162])




```python
def f_jacobian(x):
    return [[-3*x[0]**2-4*x[0], 1], [2*x[0], 1]]
```


```python
optimize.fsolve(f, [1, 1], fprime=f_jacobian)
```




    array([0.73205081, 0.46410162])




```python
x, y = sympy.symbols("x, y")

f_mat = sympy.Matrix([y - x**3 -2*x**2 + 1, y + x**2 - 1])
f_mat.jacobian(sympy.Matrix([x, y]))
```




$\displaystyle \left[\begin{matrix}- 3 x^{2} - 4 x & 1\\2 x & 1\end{matrix}\right]$



* Below: how three solutions can be found to the given equation system by using different initial guesses with __optimize.fsolve__.


```python
#def f(x):
#    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]

x = np.linspace(-3, 2, 5000)
y1 = x**3 + 2 * x**2 -1
y2 = -x**2 + 1

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, y1, 'b', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, 'g', lw=1.5, label=r'$y = -x^2 + 1$')

x_guesses = [[-2, 2], [1, -1], [-2, -5]]
for x_guess in x_guesses:
    sol = optimize.fsolve(f, x_guess)
    ax.plot(sol[0], sol[1], 'r*', markersize=15)

    ax.plot(x_guess[0], x_guess[1], 'ko')
    ax.annotate("", xy=(sol[0], sol[1]), xytext=(x_guess[0], x_guess[1]),
                arrowprops=dict(arrowstyle="->", linewidth=2.5))
    
ax.legend(loc=0)
ax.set_xlabel(r'$x$', fontsize=18)
fig.tight_layout()
fig.savefig('ch5-nonlinear-system.pdf')
```


    
![png](ch05-equation-solving_files/ch05-equation-solving_81_0.png)
    



```python
optimize.broyden2(f, x_guesses[1])
```




    array([0.73205079, 0.46410162])



* Below: using visualization to see how different initial guesses converge to different solutions. 
* Each dot represents initial guess; color indicates which solution (out of 3) it eventually converges into.
* Missing "dots": initial guesses that cannot be converged to a solution.


```python
def f(x):
    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1,
            x[1] + x[0]**2 - 1]

x = np.linspace(-3, 2, 5000)
y1 = x**3 + 2 * x**2 -1
y2 = -x**2 + 1

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, y1, 'k', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, 'k', lw=1.5, label=r'$y = -x^2 + 1$')

sol1 = optimize.fsolve(f, [-2,  2])
sol2 = optimize.fsolve(f, [ 1, -1])
sol3 = optimize.fsolve(f, [-2, -5])

colors = ['r', 'b', 'g']
for m in np.linspace(-4, 3, 80):
    for n in np.linspace(-15, 15, 40):
        x_guess = [m, n]
        sol = optimize.fsolve(f, x_guess)

        for idx, s in enumerate([sol1, sol2, sol3]):
            if abs(s-sol).max() < 1e-8:
                ax.plot(sol[0], sol[1], colors[idx]+'*', markersize=15)
                ax.plot(x_guess[0], x_guess[1], colors[idx]+'.')
    
ax.set_xlabel(r'$x$', fontsize=18)
fig.tight_layout()
fig.savefig('ch5-nonlinear-system-map.pdf')
```

    /usr/lib/python3/dist-packages/scipy/optimize/_minpack_py.py:178: RuntimeWarning: The iteration is not making good progress, as measured by the 
      improvement from the last ten iterations.
      warnings.warn(msg, RuntimeWarning)



    
![png](ch05-equation-solving_files/ch05-equation-solving_84_1.png)
    



