# Interpolation
* How to construct a function from discrete set of data points.
* Similar to least square fit, but goal is to find function that **exactly** matches given data points.

- __Introduction__
- __Polynomials__
- __Polynomial Interpolation__
- __Spline Interpolation__
- __Multivariate Interpolation__


```python
%matplotlib inline

import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = "12"

import matplotlib.pyplot as plt
```


```python
import numpy as np
from numpy import polynomial as P
from scipy import interpolate
from scipy import linalg
```

### Working with Polynomials
* __numpy.poly1d__ and __numpy.polynomial__ have large functional overlap, but incompatible. Use __numpy.polynomial__ for all new code.


```python
# to create polynomial 1 + 2*x + 3*x^2
p1 = P.Polynomial([1,2,3])

# alternative create method by specifying roots
p2 = P.Polynomial.fromroots([-1, 1])

print(p1,"\n",p2)
```

    1.0 + 2.0·x + 3.0·x² 
     -1.0 + 0.0·x + 1.0·x²



```python
# computing roots
print(p1.roots(),"\n",p2.roots())
```

    [-0.33333333-0.47140452j -0.33333333+0.47140452j] 
     [-1.  1.]


* domain & window attributes - used to map input to another interval
* best use case: working with polynomials orthogonal to a scalar product


```python
print(p1.coef,"\n",p1.domain,"\n",p1.window)
```

    [1. 2. 3.] 
     [-1  1] 
     [-1  1]



```python
# makes polynomial evaluation with arbitrary args rather easy.
p1(np.array([1.5, 2.5, 3.5]))
```




    array([10.75, 24.75, 44.75])



* Instances of __Polynomial__ can be used for std math operations.


```python
p1+p2
```




$x \mapsto \color{LightGray}{\text{0.0}} + \text{2.0}\,x + \text{4.0}\,x^{2}$




```python
p2/5
```




$x \mapsto \text{-0.2}\color{LightGray}{ + \text{0.0}\,x} + \text{0.2}\,x^{2}$




```python
p1 = P.Polynomial.fromroots([1, 2, 3]); p1
```




$x \mapsto \text{-6.0} + \text{11.0}\,x - \text{6.0}\,x^{2} + \text{1.0}\,x^{3}$




```python
p2 = P.Polynomial.fromroots([2]); p2
```




$x \mapsto \text{-2.0} + \text{1.0}\,x$




```python
p3 = p1 // p2; p3
```




$x \mapsto \text{3.0} - \text{4.0}\,x + \text{1.0}\,x^{2}$




```python
p3.roots()
```




    array([1., 3.])



* Polynomial offers support for __Chebyshev__, __Legendre__, __Laguerre__ & __Hermite__ bases.


```python
c1 = P.Chebyshev([1, 2, 3]); c1
```




$x \mapsto \text{1.0}\,{T}_{0}(x) + \text{2.0}\,{T}_{1}(x) + \text{3.0}\,{T}_{2}(x)$




```python
c1.roots()
```




    array([-0.76759188,  0.43425855])




```python
c = P.Chebyshev.fromroots([-1, 1]); c
```




$x \mapsto \text{-0.5}\,{T}_{0}(x)\color{LightGray}{ + \text{0.0}\,{T}_{1}(x)} + \text{0.5}\,{T}_{2}(x)$




```python
l = P.Legendre.fromroots([-1, 1]); l
```




$x \mapsto \text{-0.66666667}\,{P}_{0}(x)\color{LightGray}{ + \text{0.0}\,{P}_{1}(x)} + \text{0.66666667}\,{P}_{2}(x)$




```python
c1(np.array([0.5, 1.5, 2.5]))
```




    array([ 0.5, 14.5, 40.5])




```python
l(np.array([0.5, 1.5, 2.5]))
```




    array([-0.75,  1.25,  5.25])



### Polynomial interpolation


```python
x = np.array([1, 2, 3, 4])
y = np.array([1, 3, 5, 4])
```


```python
# to interpolate polynomial thru points above, we need a
# polynomial of 3rd degree (#datapoints -1).
deg = len(x) - 1
deg
```




    3




```python
# find Vandermonde matrix
A = P.polynomial.polyvander(x, deg)
```


```python
# solve interpolation, returns coefficients vector
c = linalg.solve(A, y)
c
```




    array([ 2. , -3.5,  3. , -0.5])




```python
f1 = P.Polynomial(c)
f1(2.5)
```




    4.1875




```python
f1(2.5)
```




    4.1875




```python
# polynomial interpolation in another basis == change function
# name used to generate Vandermonde matrix
A = P.chebyshev.chebvander(x, deg)
```


```python
c = linalg.solve(A, y)
c
```




    array([ 3.5  , -3.875,  1.5  , -0.125])




```python
f2 = P.Chebyshev(c)
f2(2.5)
```




    4.1875




```python
xx = np.linspace(x.min(), x.max(), 100)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(xx, f1(xx), 'b', lw=2, label='Power basis interp.')
ax.plot(xx, f2(xx), 'r--', lw=2, label='Chebyshev basis interp.')
ax.scatter(x, y, label='data points')

ax.legend(loc=4)
ax.set_xticks(x)
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-polynomial-interpolation.pdf');
```


    
![png](ch07-interpolation_files/ch07-interpolation_33_0.png)
    



```python
# better method: using generalized fit method
f1b = P.Polynomial.fit(x, y, deg)
f1b
```




$x \mapsto \text{4.1875} + \text{3.1875}\,\left(\text{-1.66666667} + \text{0.66666667}x\right) - \text{1.6875}\,\left(\text{-1.66666667} + \text{0.66666667}x\right)^{2} - \text{1.6875}\,\left(\text{-1.66666667} + \text{0.66666667}x\right)^{3}$




```python
f2b = P.Chebyshev.fit(x, y, deg)
f2b
```




$x \mapsto \text{3.34375}\,{T}_{0}(\text{-1.66666667} + \text{0.66666667}x) + \text{1.921875}\,{T}_{1}(\text{-1.66666667} + \text{0.66666667}x) - \text{0.84375}\,{T}_{2}(\text{-1.66666667} + \text{0.66666667}x) - \text{0.421875}\,{T}_{3}(\text{-1.66666667} + \text{0.66666667}x)$




```python
np.linalg.cond(P.chebyshev.chebvander(x, deg))
```




    4659.738424139918




```python
np.linalg.cond(P.chebyshev.chebvander((2*x-5)/3.0, deg))
```




    1.8542033440472891




```python
(2 * x - 5)/3.0
```




    array([-1.        , -0.33333333,  0.33333333,  1.        ])




```python
f1 = P.Polynomial.fit(x, y, 1)
f2 = P.Polynomial.fit(x, y, 2)
f3 = P.Polynomial.fit(x, y, 3)
```


```python
xx = np.linspace(x.min(), x.max(), 100)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(xx, f1(xx), 'r', lw=2, label='1st order')
ax.plot(xx, f2(xx), 'g', lw=2, label='2nd order')
ax.plot(xx, f3(xx), 'b', lw=2, label='3rd order')
ax.scatter(x, y, label='data points')

ax.legend(loc=4)
ax.set_xticks(x)
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18);
```


    
![png](ch07-interpolation_files/ch07-interpolation_40_0.png)
    


### Runge problem
* When #data points increase, we need higher-order polynomials for exact interpolation == problematic. (Example: wide variation between specified datapoints. Runge's function == illustration.)


```python
def runge(x):
    return 1/(1 + 25 * x**2)
```


```python
def runge_interpolate(n):
    x = np.linspace(-1, 1, n+1)
    p = P.Polynomial.fit(x, runge(x), deg=n)
    return x, p
```


```python
xx = np.linspace(-1, 1, 250)
```


```python
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(xx, runge(xx), 'k', lw=2, label="Runge's function")

n = 13
x, p = runge_interpolate(n)
ax.plot(x, runge(x), 'ro')
ax.plot(xx, p(xx), 'r', label='interp. order %d' % n)

n = 14
x, p = runge_interpolate(n)
ax.plot(x, runge(x), 'go')
ax.plot(xx, p(xx), 'g', label='interp. order %d' % n)

ax.legend(loc=8)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1, 2)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-polynomial-interpolation-runge.pdf');
```


    
![png](ch07-interpolation_files/ch07-interpolation_45_0.png)
    


### Spline interpolation
* Piecewise polynomial of degree k == spline if it continuously differentiable k-1 times.


```python
x = np.linspace(-1, 1, 11)
```


```python
y = runge(x)
```


```python
f = interpolate.interp1d(x, y, kind=3)
```


```python
xx = np.linspace(-1, 1, 100)
```


```python
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(xx, runge(xx), 'k', lw=1, label="Runge's function")
ax.plot(x, y, 'ro', label='sample points')
ax.plot(xx, f(xx), 'r--', lw=2, label='spline order 3')

ax.legend()
ax.set_ylim(0, 1.1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-spline-interpolation-runge.pdf');
```


    
![png](ch07-interpolation_files/ch07-interpolation_51_0.png)
    



```python
# to illustrate order effects of a spline interpolation:

x = np.array([0, 1, 2,   3, 4, 5,   6,    7])
y = np.array([3, 4, 3.5, 2, 1, 1.5, 1.25, 0.9])
```


```python
xx = np.linspace(x.min(), x.max(), 100)
```


```python
fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(x, y)

for n in [1, 2, 3, 5]:
    f = interpolate.interp1d(x, y, kind=n)
    ax.plot(xx, f(xx), label='order %d' % n)

ax.legend()
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-spline-interpolation-orders.pdf');

# higher-order splines starting to diverge
```


    
![png](ch07-interpolation_files/ch07-interpolation_54_0.png)
    


### Multivariate interpolation
- useful functions: __interpolate.interp2d__ and __interpolate.griddata__.


```python
x = y = np.linspace(-2, 2, 10)
```


```python
def f(x, y):
    return np.exp(-(x + .5)**2 - 2*(y + .5)**2) - np.exp(-(x - .5)**2 - 2*(y - .5)**2)
```


```python
X, Y = np.meshgrid(x, y)
```


```python
# simulate noisy data at fixed grid points X, Y
Z = f(X, Y) + 0.05 * np.random.randn(*X.shape)
```


```python
f_interp = interpolate.interp2d(x, y, Z, kind='cubic')
```

    /tmp/ipykernel_65539/2525062336.py:1: DeprecationWarning: `interp2d` is deprecated!
    `interp2d` is deprecated in SciPy 1.10 and will be removed in SciPy 1.12.0.
    
    For legacy code, nearly bug-for-bug compatible replacements are
    `RectBivariateSpline` on regular grids, and `bisplrep`/`bisplev` for
    scattered 2D data.
    
    In new code, for regular grids use `RegularGridInterpolator` instead.
    For scattered data, prefer `LinearNDInterpolator` or
    `CloughTocher2DInterpolator`.
    
    For more details see
    `https://gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff`
    
      f_interp = interpolate.interp2d(x, y, Z, kind='cubic')



```python
xx = yy = np.linspace(x.min(), x.max(), 100)
```


```python
ZZi = f_interp(xx, yy)
```

    /tmp/ipykernel_65539/408824875.py:1: DeprecationWarning:         `interp2d` is deprecated!
            `interp2d` is deprecated in SciPy 1.10 and will be removed in SciPy 1.12.0.
    
            For legacy code, nearly bug-for-bug compatible replacements are
            `RectBivariateSpline` on regular grids, and `bisplrep`/`bisplev` for
            scattered 2D data.
    
            In new code, for regular grids use `RegularGridInterpolator` instead.
            For scattered data, prefer `LinearNDInterpolator` or
            `CloughTocher2DInterpolator`.
    
            For more details see
            `https://gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff`
    
      ZZi = f_interp(xx, yy)



```python
XX, YY = np.meshgrid(xx, yy)
```


```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

c = axes[0].contourf(XX, YY, f(XX, YY), 15, cmap=plt.cm.RdBu)
axes[0].set_xlabel(r"$x$", fontsize=20)
axes[0].set_ylabel(r"$y$", fontsize=20)
axes[0].set_title("exact / high sampling")
cb = fig.colorbar(c, ax=axes[0])
cb.set_label(r"$z$", fontsize=20)

c = axes[1].contourf(XX, YY, ZZi, 15, cmap=plt.cm.RdBu)
axes[1].set_ylim(-2.1, 2.1)
axes[1].set_xlim(-2.1, 2.1)
axes[1].set_xlabel(r"$x$", fontsize=20)
axes[1].set_ylabel(r"$y$", fontsize=20)
axes[1].scatter(X, Y, marker='x', color='k')
axes[1].set_title("interpolation of noisy data / low sampling")
cb = fig.colorbar(c, ax=axes[1])
cb.set_label(r"$z$", fontsize=20)

fig.tight_layout()
fig.savefig('ch7-multivariate-interpolation-regular-grid.pdf')
```


    
![png](ch07-interpolation_files/ch07-interpolation_64_0.png)
    



```python
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

c = ax.contourf(XX, YY, ZZi, 15, cmap=plt.cm.RdBu)
ax.set_ylim(-2.1, 2.1)
ax.set_xlim(-2.1, 2.1)
ax.set_xlabel(r"$x$", fontsize=20)
ax.set_ylabel(r"$y$", fontsize=20)
ax.scatter(X, Y, marker='x', color='k')
cb = fig.colorbar(c, ax=ax)
cb.set_label(r"$z$", fontsize=20)

fig.tight_layout()
#fig.savefig('ch7-multivariate-interpolation-regular-grid.pdf')
```


    
![png](ch07-interpolation_files/ch07-interpolation_65_0.png)
    


### Irregular grid

* common use case requiring multivariate interpolation: when sampled data comes from an irregular coordinate grid.
* SciPy's __interpolate.griddata__ function maps irregular data to a regular grid.
* It accepts 'nearest','linear','cubic' interpolation arguments.


```python
np.random.seed(115925231)
```


```python
x = y = np.linspace(-1, 1, 100)
```


```python
X, Y = np.meshgrid(x, y)
```


```python
def f(x, y):
    return np.exp(-x**2 - y**2) * np.cos(4*x) * np.sin(6*y)
```


```python
Z = f(X, Y)
```


```python
N = 500
```


```python
xdata = np.random.uniform(-1, 1, N)
ydata = np.random.uniform(-1, 1, N)
zdata = f(xdata, ydata)
```


```python
#contour plot, randomly sampled function (n=500)

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(X, Y, Z, 15, cmap=plt.cm.RdBu);
ax.scatter(xdata, ydata, marker='.')
ax.set_ylim(-1,1)
ax.set_xlim(-1,1)
ax.set_xlabel(r"$x$", fontsize=20)
ax.set_ylabel(r"$y$", fontsize=20)

cb = fig.colorbar(c, ax=ax)
cb.set_label(r"$z$", fontsize=20)

fig.tight_layout()
fig.savefig('ch7-multivariate-interpolation-exact.pdf');
```


    
![png](ch07-interpolation_files/ch07-interpolation_74_0.png)
    



```python
# helper function - interpolates data pts with three methods
def z_interpolate(xdata, ydata, zdata):
    Zi_0 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='nearest')
    Zi_1 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='linear')
    Zi_3 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='cubic')
    return Zi_0, Zi_1, Zi_3
```


```python
fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

n_vec = [50, 150, 500]

for idx, n in enumerate(n_vec):
    Zi_0, Zi_1, Zi_3 = z_interpolate(xdata[:n], ydata[:n], zdata[:n])
    axes[idx, 0].contourf(X, Y, Zi_0, 15, cmap=plt.cm.RdBu)
    axes[idx, 0].set_ylabel("%d data points\ny" % n, fontsize=16)
    axes[idx, 0].set_title("nearest", fontsize=16)
    axes[idx, 1].contourf(X, Y, Zi_1, 15, cmap=plt.cm.RdBu)
    axes[idx, 1].set_title("linear", fontsize=16)
    axes[idx, 2].contourf(X, Y, Zi_3, 15, cmap=plt.cm.RdBu)
    axes[idx, 2].set_title("cubic", fontsize=16)

for m in range(len(n_vec)):
    axes[idx, m].set_xlabel("x", fontsize=16)
    
fig.tight_layout()
fig.savefig('ch7-multivariate-interpolation-interp.pdf');
```


    
![png](ch07-interpolation_files/ch07-interpolation_76_0.png)
    





