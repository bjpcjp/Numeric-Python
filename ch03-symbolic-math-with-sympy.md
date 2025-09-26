# Symbolic computing with [SymPy](https://www.sympy.org/en/index.html)

Symbolic analysis can make a difference in determining __how__ to attack a problem before throwing numerical firepower at it. (Complexity or size reduction, for example.)

- __Symbols__
- __Numbers__ (Integer, Float, Rational)
- __Constants__
- __Functions__
- __Expression__
- __Manipulating expressions__
- __Simplification__
- __Expansion__
- __Factor, Collect, Combine__
- __Apart, Together, Cancel__
- __Substitutions__
- __Numerical evaluation__
- __Calculus__ (Derivatives, Integrals, Series, Limits)
- __Sums & Products__
- __Equations__
- __Linear Algebra__


```python
import sympy
from sympy import I, pi, oo # frequently used symbols
```


```python
sympy.init_printing() # enables MathJax to render SymPy expressions
```

## Symbols

Symbols have a name & a set of attributes. They are not especially useful by themselves, but are used as nodes in algebraic expression trees.

![symbol arguments](pics/symbol-arguments.png)

Symbols are used as nodes in algebraic expression trees.
Normally, symbols are associated with Python variables with similar names.


```python
print(sympy.Symbol("x"               ).is_real)
print(sympy.Symbol('y',     real=True).is_real)
print(sympy.Symbol("z",imaginary=True).is_real)
```

    None
    True
    False


- Below: Sympy recognizes $\sqrt{y^2} = y$


```python
x = sympy.Symbol("x")
y = sympy.Symbol("y", positive=True)
sympy.sqrt(x**2), sympy.sqrt(y**2)
```




$\displaystyle \left( \sqrt{x^{2}}, \  y\right)$



Integer representation


```python
n1 = sympy.Symbol("n")
n2 = sympy.Symbol("n", integer=True)
n3 = sympy.Symbol("n", odd=True)

sympy.cos(n1*pi), sympy.cos(n2*pi), sympy.cos(n3*pi)
```




$\displaystyle \left( \cos{\left(\pi n \right)}, \  \left(-1\right)^{n}, \  -1\right)$



Creating multiple symbols in one function call


```python
a, b, c = sympy.symbols("a, b, c", negative=True)
d, e, f = sympy.symbols("d, e, f", positive=True)
```

### Numbers
Can't directly use Python objects for integers, floats.

Instead use __Sympy.Integer__, __Sympy.Float__ (Not often needed, because Sympy auto-promotes numbers to class instances when needed.)


```python
i = sympy.Integer(19)
i.is_Integer, i.is_real, i.is_odd
```




    (True, True, True)




```python
f = sympy.Float(2.3)
f.is_Integer, f.is_real, f.is_odd
```




    (False, True, False)




```python
i, f = sympy.sympify(19), sympy.sympify(2.3)
type(i), type(f)
```




    (sympy.core.numbers.Integer, sympy.core.numbers.Float)



### Integers

There is a difference between a `Symbol instance` with `integer=True` and an instance of Integer. `Symbol` with `integer=True` represents some integer; an Integer instance represents a specific integer. 


`is_integer` is True for both. `is_Integer` (note the capital I) is only True for Integer instances. 

SymPy integers & floats have arbitrary precision, no upper/lower bounds. They are much easier to use with very large numbers.


```python
n = sympy.Symbol("n", integer=True)
i = sympy.Integer(19)
```


```python
print(n.is_integer, n.is_Integer, n.is_positive, n.is_Symbol)
print(i.is_integer, i.is_Integer, i.is_positive, i.is_Symbol)
```

    True False None True
    True True True False



```python
print(i**50)
```

    8663234049605954426644038200675212212900743262211018069459689001



```python
print(sympy.factorial(100))
```

    93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000


### Float

Also arbitrary precision. SymPy Float can represent the real number 0.3 without the limitations of floating-point format.


```python
"%.25f" % 0.3  # create a string represention with 25 decimals
```




    '0.2999999999999999888977698'




```python
print(sympy.Float( 0.3, 25))
print(sympy.Float('0.3',25))
```

    0.2999999999999999888977698
    0.3000000000000000000000000


### Rationals

Fractions of two integers.


```python
sympy.Rational(11, 13)
```




$\displaystyle \frac{11}{13}$




```python
r1 = sympy.Rational(2, 3)
r2 = sympy.Rational(4, 5)
r1*r2, r1/r2
```




$\displaystyle \left( \frac{8}{15}, \  \frac{5}{6}\right)$



### Constants & Special Symbols
![constants-specials](pics/constants-specials.png)


```python
sympy.pi, sympy.E, sympy.EulerGamma, sympy.I, sympy.oo
```




$\displaystyle \left( \pi, \  e, \  \gamma, \  i, \  \infty\right)$



### Functions

Sympy understands __defined__ vs __undefined__ functions, and __applied__ vs __unapplied__ functions.

Creating a function with __Function__ returns an undefined, unapplied function. It has a name but cannot be evaluated.


```python
f = sympy.Function("f"); print(type(f))
```

    <class 'sympy.core.function.UndefinedFunction'>



```python
f(x)
```




$\displaystyle f{\left(x \right)}$




```python
x,y,z = sympy.symbols("x, y, z")
g     = sympy.Function("g")(x,y,z)
g, g.free_symbols
```




$\displaystyle \left( g{\left(x,y,z \right)}, \  \left\{x, y, z\right\}\right)$




```python
# defined functions have a specific implementation.
# they can be numerically evaluated.
sympy.sin, sympy.sin(x), sympy.sin(pi*1.5)
```




    (sin, sin(x), -1)




```python
n = sympy.Symbol("n", integer=True)
sympy.sin(n*pi)
```




$\displaystyle 0$




```python
# lambda functions: no name, but an executable body.
h = sympy.Lambda(x, x**2)
h, h(5), h(1+x)
```




$\displaystyle \left( \left( x \mapsto x^{2} \right), \  25, \  \left(x + 1\right)^{2}\right)$



### Expressions
Sympy expressions = represented as tree structures.

Symbols = leaves; nodes = math op class instances.


```python
x = sympy.Symbol("x")
e = 1 + 2*x**2 + 3*x**3
e, e.args, e.args[1]
```




$\displaystyle \left( 3 x^{3} + 2 x^{2} + 1, \  \left( 1, \  2 x^{2}, \  3 x^{3}\right), \  2 x^{2}\right)$



![expression tree](pics/expression-tree.png)


```python
# explore the expression tree using the args attribute
# args is a tuple of sub-expressions
e.args[1].args[1], e.args[1].args[0], e.args[2].args[1]
```




$\displaystyle \left( x^{2}, \  2, \  x^{3}\right)$



### Manipulating Expressions

SymPy's main job to provide different transforms to expression trees. These transforms create new expressions & do not change the originals. In other words, *expression trees are immutable*.

### Simplifications

Simplification is a _good thing_, but often very ambiguous. (It's often non-trivial to determine if an expression appears simplier to a human being.)

![simplifications](pics/simplifications.png)


```python
expr = 2*(x**2-x) - x*(x+1); expr
```




$\displaystyle 2 x^{2} - x \left(x + 1\right) - 2 x$




```python
sympy.simplify(expr), expr.simplify()
```




$\displaystyle \left( x \left(x - 3\right), \  x \left(x - 3\right)\right)$



Both `sympy.simplify(expr)` and `expr.simplify()` return new
expression trees and leave the expression expr untouched. 

In this example, `expr` can be simplified by expanding the products, canceling terms, then refactoring. `sympy.simplify` will attempt
different strategies and will simplify trigonometric and
power expressions.


```python
expr = 2 * sympy.cos(x) * sympy.sin(x)
expr, sympy.simplify(expr)
```




$\displaystyle \left( 2 \sin{\left(x \right)} \cos{\left(x \right)}, \  \sin{\left(2 x \right)}\right)$




```python
expr = sympy.exp(x) * sympy.exp(y)
expr, sympy.simplify(expr)

# can also use sympy.trigsimp and sympy.powsimp 
# to perform only their specified simplifications
# leaving the rest of the expression untouched.
```




$\displaystyle \left( e^{x} e^{y}, \  e^{x + y}\right)$



### Expansion

When __sympy.simplify__ does not provide acceptable results. Expression expansion can help with designing a more manual approach.


```python
expr = (x+1)*(x+2)
sympy.expand(expr)
```




$\displaystyle x^{2} + 3 x + 2$




```python
# trig expansions
sympy.sin(x+y).expand(trig=True) 
```




$\displaystyle \sin{\left(x \right)} \cos{\left(y \right)} + \sin{\left(y \right)} \cos{\left(x \right)}$




```python
# logarithmic expansions
a, b = sympy.symbols("a, b", positive=True)
sympy.log(a*b).expand(log=True) 
```




$\displaystyle \log{\left(a \right)} + \log{\left(b \right)}$




```python
# separating real & imag parts
sympy.exp(I*a + b).expand(complex=True) 
```




$\displaystyle i e^{b} \sin{\left(a \right)} + e^{b} \cos{\left(a \right)}$




```python
# power expressions - expanding the base & exponent
sympy.expand((a*b)**x, power_exp=True) 
```




$\displaystyle a^{x} b^{x}$




```python
sympy.exp(I*(a-b)*x).expand(power_exp=True)
```




$\displaystyle e^{i a x} e^{- i b x}$



### Factoring, Collecting, Combining

__expand__ often used to expand a function, cancel some terms, then refactor or recombine the expression. __factor__ helps do this.


```python
sympy.factor(x**2 - 1)
```




$\displaystyle \left(x - 1\right) \left(x + 1\right)$




```python
sympy.factor(x*sympy.cos(y) + x*sympy.sin(z))
```




$\displaystyle x \left(\sin{\left(z \right)} + \cos{\left(y \right)}\right)$




```python
sympy.logcombine(sympy.log(a) - sympy.log(b))
```




$\displaystyle \log{\left(\frac{a}{b} \right)}$




```python
# use collect for fine-grained factor control
# collect factors terms containing a given symbol or list of symbols.
expr = x + y + x*y*z
expr.factor(), expr.collect(x), expr.collect(y)
```




$\displaystyle \left( x y z + x + y, \  x \left(y z + 1\right) + y, \  x + y \left(x z + 1\right)\right)$




```python
# collect also supports method chaining
expr = sympy.cos(x+y) + sympy.sin(x-y)

expr.expand(trig=True).collect(
    [sympy.cos(x), 
     sympy.sin(x)]).collect(sympy.cos(y) - sympy.sin(y))
```




$\displaystyle \left(\sin{\left(x \right)} + \cos{\left(x \right)}\right) \left(- \sin{\left(y \right)} + \cos{\left(y \right)}\right)$



## Together, apart, cancel - rewriting of fractions


```python
# rewrite fraction as a partial fraction
sympy.apart(1/(x**2 + 3*x + 2), x) 
```




$\displaystyle - \frac{1}{x + 2} + \frac{1}{x + 1}$




```python
# combine partial into single fraction
sympy.together(1/(y*x+y) + 1/(1+x)) 
```




$\displaystyle \frac{y + 1}{y \left(x + 1\right)}$




```python
# cancel shared factors btwn numerator, denominator
sympy.cancel(y/(y*x+y)) 
```




$\displaystyle \frac{1}{x + 1}$



## Substitutions

Th most basic use case: method called on an expression. The 1st argument = symbol/expr to be replaced; the 2nd argument = new symbol/expr.


```python
(x+y).subs(x,y)
```




$\displaystyle 2 y$




```python
sympy.sin(x*sympy.exp(x)).subs(x,y)
```




$\displaystyle \sin{\left(y e^{y} \right)}$



Instead of chaining multiple __subs__ calls when multiple substitutions are required, we can pass a dictionary as the only argument. It maps old symbols or expressions to new ones.


```python
sympy.sin(x*z).subs({z: sympy.exp(y), x: y, sympy.sin: sympy.cos})
```




$\displaystyle \cos{\left(y e^{y} \right)}$



Typical use case: substitute numerical values for symbolic numbers. A convenient way of doing this is to define a dictionary that translates the symbols to numerical values, and passing this dictionary to the subs method.


```python
expr = x*y + z**2*x
values = {x: 1.25, y: 0.4, z: 3.2}
expr.subs(values)
```




$\displaystyle 13.3$



## Numerical evaluation

Sooner or later you'll need to evaluate symbolic expressions numerically (for plots or  numerical results). A SymPy expression can be evaluated using either __sympy.N__ or the __evalf__ method.


```python
# 2nd argument = #significant digits
sympy.N(1+pi), sympy.N(pi, 20)
```




$\displaystyle \left( 4.14159265358979, \  3.1415926535897932385\right)$




```python
(x + 1/pi).evalf(10)
```




$\displaystyle x + 0.3183098862$



When we need to evaluate an expression for multiple inputs, we can loop the evaluation.


```python
expr = sympy.sin(pi*x*sympy.exp(x))
[expr.subs(x, xx).evalf(3) for xx in range(0, 10)]
```




$\displaystyle \left[ 0, \  0.774, \  0.642, \  0.722, \  0.944, \  0.205, \  0.974, \  0.977, \  -0.87, \  -0.695\right]$



Looping is slow. SymPy provides __sympy.lambdify__ instead. It takes a set of free symbols and an expression, and generates a function that evaluates the value of the expression.

The function takes the same number of arguments as the number of free symbols passed as first argument to sympy.lambdify.


```python
expr_func = sympy.lambdify(x, expr)
expr_func(1.0)
```




$\displaystyle 0.773942685266709$



__expr_func__ expects numerical (scalar) values, so we can't pass a symbol. SymPy can generate functions that are NumPy-aware. Below: How a SymPy expression is converted to NumPy-aware vectorized function for efficient evaluation.


```python
expr_func = sympy.lambdify(x, expr, 'numpy')
expr_func(1.0)
```




$\displaystyle 0.773942685266709$




```python
import numpy as np
xvalues = np.arange(0, 10)
expr_func(xvalues)
```




    array([ 0.        ,  0.77394269,  0.64198244,  0.72163867,  0.94361635,
            0.20523391,  0.97398794,  0.97734066, -0.87034418, -0.69512687])



## Calculus - Derivatives

The derivative of a function describes a rate of change at a given point. We can use __sympy.diff__ or the __diff__ method of SymPy expressions. These functions accept a symbol or symbols for which the function or the expression is to be derived with respect to.


```python
f = sympy.Function('f')(x)
```


```python
# first-order derivatives
sympy.diff(f,x), sympy.diff(f,x,x), sympy.diff(f,x,3)
```




$\displaystyle \left( \frac{d}{d x} f{\left(x \right)}, \  \frac{d^{2}}{d x^{2}} f{\left(x \right)}, \  \frac{d^{3}}{d x^{3}} f{\left(x \right)}\right)$




```python
# multivariate function derivatives
g = sympy.Function('g')(x,y)
g.diff(x,y), g.diff(x,3,y,2)
```




$\displaystyle \left( \frac{\partial^{2}}{\partial y\partial x} g{\left(x,y \right)}, \  \frac{\partial^{5}}{\partial y^{2}\partial x^{3}} g{\left(x,y \right)}\right)$



We can evaluate derivatives of defined functions, which result in new expressions that correspond to the evaluated derivatives. Below: using sympy.diff to evaluate derivatives of arbitrary mathematical expressions, such as polynomials.


```python
expr = x**4 + x**3 + x**2 + x+1
expr.diff(x), expr.diff(x,x)
```




$\displaystyle \left( 4 x^{3} + 3 x^{2} + 2 x + 1, \  2 \cdot \left(6 x^{2} + 3 x + 1\right)\right)$




```python
expr = (x+1)**3 * y**2 * (z-1)
expr.diff(x, y, z)
```




$\displaystyle 6 y \left(x + 1\right)^{2}$




```python
# trig function differentials
expr = sympy.sin(x*y) * sympy.cos(x/2)
expr.diff(x)
```




$\displaystyle y \cos{\left(\frac{x}{2} \right)} \cos{\left(x y \right)} - \frac{\sin{\left(\frac{x}{2} \right)} \sin{\left(x y \right)}}{2}$




```python
# special function differentials (updated: formerly "sympy.special...")
expr = sympy.functions.special.polynomials.hermite(x,0)
expr.diff(x).doit()
```




$\displaystyle \frac{2^{x} \sqrt{\pi} \operatorname{polygamma}{\left(0,\frac{1}{2} - \frac{x}{2} \right)}}{2 \Gamma\left(\frac{1}{2} - \frac{x}{2}\right)} + \frac{2^{x} \sqrt{\pi} \log{\left(2 \right)}}{\Gamma\left(\frac{1}{2} - \frac{x}{2}\right)}$



Calling sympy.diff results in a new expression.

If instead we want to represent a derivative of a definite expression, create an instance of the Derivative class. 1st argument = the expression; 2nd argument = the derivative.

This (formal) representation can be evaluated using __doit__.


```python
d = sympy.Derivative(sympy.exp(sympy.cos(x)), x)
d, d.doit()
```




$\displaystyle \left( \frac{d}{d x} e^{\cos{\left(x \right)}}, \  - e^{\cos{\left(x \right)}} \sin{\left(x \right)}\right)$



## Integrals

Integrals are evaluated using __sympy.integrate__.

Formal integrals represented using __sympy.Integral__.

Two forms: __definite__ (specified integration limits, can be interpreted as an area or volume) & __indefinite__ (no integration limits)


```python
a, b = sympy.symbols("a, b")
x, y = sympy.symbols('x, y')
f    = sympy.Function('f')(x)
```


```python
sympy.integrate(f), sympy.integrate(f,(x,a,b))
```




$\displaystyle \left( \int f{\left(x \right)}\, dx, \  \int\limits_{a}^{b} f{\left(x \right)}\, dx\right)$




```python
sympy.integrate(sympy.sin(x))
```




$\displaystyle - \cos{\left(x \right)}$




```python
sympy.integrate(sympy.sin(x), (x,a,b))
```




$\displaystyle \cos{\left(a \right)} - \cos{\left(b \right)}$






```python
# Definite integrals can use limits from negative infinity to positive infinity.
sympy.integrate(sympy.exp(-x**2), (x,0,oo))
```




$\displaystyle \frac{\sqrt{\pi}}{2}$




```python
a,b,c = sympy.symbols("a, b, c", positive=True)
```


```python
sympy.integrate(a * sympy.exp(-((x-b)/c)**2), (x, -oo, oo))
```




$\displaystyle \sqrt{\pi} a c$



Computing integrals symbolically is a difficult problem. When SymPy fails to evaluate an integral, an instance of __sympy.Integral__, representing the formal integral, is returned instead.


```python
sympy.integrate(sympy.sin(x * sympy.cos(x)))
```




$\displaystyle \int \sin{\left(x \cos{\left(x \right)} \right)}\, dx$



Multivariable expressions can be integrated. In the case of indefinite integral of a multivariable expression, the integration variable has to be specified explicitly.


```python
expr = sympy.sin(x*sympy.exp(y))
sympy.integrate(expr, x)
```




$\displaystyle - e^{- y} \cos{\left(x e^{y} \right)}$




```python
expr = (x+y)**2
sympy.integrate(expr,x)
```




$\displaystyle \frac{x^{3}}{3} + x^{2} y + x y^{2}$



By passing more than one symbol, or more than one tuple that contain symbols and their integration limits, we can carry out multiple integration.


```python
sympy.integrate(expr,x,y)
```




$\displaystyle \frac{x^{3} y}{3} + \frac{x^{2} y^{2}}{2} + \frac{x y^{3}}{3}$




```python
sympy.integrate(expr,(x,0,1),(y,0,1))
```




$\displaystyle \frac{7}{6}$



## Series expansions

With a series expansion, a function can be written as a polynomial with coefficients given by the derivatives of the function at the point where the expansion is made. By truncating the expansion at some order n, the nth order approximation of the function is obtained.


```python
# undefined functions: expansion is computed up to 6th order around x0=0.
x = sympy.Symbol("x")
f = sympy.Function("f")(x)
sympy.series(f,x)
```




$\displaystyle f{\left(0 \right)} + x \left. \frac{d}{d \xi} f{\left(\xi \right)} \right|_{\substack{ \xi=0 }} + \frac{x^{2} \left. \frac{d^{2}}{d \xi^{2}} f{\left(\xi \right)} \right|_{\substack{ \xi=0 }}}{2} + \frac{x^{3} \left. \frac{d^{3}}{d \xi^{3}} f{\left(\xi \right)} \right|_{\substack{ \xi=0 }}}{6} + \frac{x^{4} \left. \frac{d^{4}}{d \xi^{4}} f{\left(\xi \right)} \right|_{\substack{ \xi=0 }}}{24} + \frac{x^{5} \left. \frac{d^{5}}{d \xi^{5}} f{\left(\xi \right)} \right|_{\substack{ \xi=0 }}}{120} + O\left(x^{6}\right)$




```python
# to specify point around which to expand, provide x0 keyword.
# n=2 tells function to return expansion up to 2nd order term.
x0 = sympy.Symbol("{x_0}")
f.series(x, x0, n=2)
```




$\displaystyle f{\left({x_0} \right)} + \left(x - {x_0}\right) \left. \frac{d}{d \xi_{1}} f{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={x_0} }} + O\left(\left(x - {x_0}\right)^{2}; x\rightarrow {x_0}\right)$




```python
# use remove0() to remove order term
f.series(x, x0, n=2).removeO()
```




$\displaystyle \left(x - {x_0}\right) \left. \frac{d}{d \xi_{1}} f{\left(\xi_{1} \right)} \right|_{\substack{ \xi_{1}={x_0} }} + f{\left({x_0} \right)}$




```python
# some well-known series expansions:
sympy.cos(x).series()
```




$\displaystyle 1 - \frac{x^{2}}{2} + \frac{x^{4}}{24} + O\left(x^{6}\right)$




```python
sympy.sin(x).series()
```




$\displaystyle x - \frac{x^{3}}{6} + \frac{x^{5}}{120} + O\left(x^{6}\right)$




```python
sympy.exp(x).series()
```




$\displaystyle 1 + x + \frac{x^{2}}{2} + \frac{x^{3}}{6} + \frac{x^{4}}{24} + \frac{x^{5}}{120} + O\left(x^{6}\right)$




```python
(1/(1+x)).series()
```




$\displaystyle 1 - x + x^{2} - x^{3} + x^{4} - x^{5} + O\left(x^{6}\right)$




```python
expr = sympy.cos(x) / (1 + sympy.sin(x * y))

expr.series(x,n=4), expr.series(y,n=4)
```




$\displaystyle \left( 1 - x y + x^{2} \left(y^{2} - \frac{1}{2}\right) + x^{3} \left(- \frac{5 y^{3}}{6} + \frac{y}{2}\right) + O\left(x^{4}\right), \  \cos{\left(x \right)} - x y \cos{\left(x \right)} + x^{2} y^{2} \cos{\left(x \right)} - \frac{5 x^{3} y^{3} \cos{\left(x \right)}}{6} + O\left(y^{4}\right)\right)$




```python
expr.series(y).removeO().series(x).removeO().expand()
```




$\displaystyle - \frac{61 x^{5} y^{5}}{120} + \frac{5 x^{5} y^{3}}{12} - \frac{x^{5} y}{24} + \frac{2 x^{4} y^{4}}{3} - \frac{x^{4} y^{2}}{2} + \frac{x^{4}}{24} - \frac{5 x^{3} y^{3}}{6} + \frac{x^{3} y}{2} + x^{2} y^{2} - \frac{x^{2}}{2} - x y + 1$



## Limits

Value of a function as a dependent variable approaches a specific value (or approaches pos/neg infinity).


```python
#To find the limit of sin(x)/x as x goes to zero:
sympy.limit(sympy.sin(x) / x, x, 0)
```




$\displaystyle 1$



We can also use __sympy.limit__ to compute symbolic limits, which can be illustrated by computing derivatives using the previous definition.


```python
f = sympy.Function('f')
x, h = sympy.symbols("x, h")
```


```python
diff_limit = (f(x+h)-f(x))/h
```


```python
sympy.limit(diff_limit.subs(f, sympy.cos), h, 0)
```




$\displaystyle - \sin{\left(x \right)}$




```python
sympy.limit(diff_limit.subs(f, sympy.sin), h, 0)
```




$\displaystyle \cos{\left(x \right)}$



Another example: find the asymptotic behavior as a function, for example
as its dependent variable approaches infinity.


```python
expr = (x**2-3*x)/(2*x-2)

p = sympy.limit(expr/x,   x, oo)
q = sympy.limit(expr-p*x, x, oo)

p,q # result: asymptotic behavior of f(x) as x becomes large:
# f(x) --> x/2-1
```




$\displaystyle \left( \frac{1}{2}, \  -1\right)$



## Sums and products

1st argument is an expression; 2nd argument is a tuple (n,n1,n2) - n is a symbol; n1 & n2 are upper & lower limits for symbol n.


```python
n = sympy.symbols("n", integer=True)
x = sympy.Sum(1/(n**2), (n, 1, oo))
```


```python
x, x.doit()
```




$\displaystyle \left( \sum_{n=1}^{\infty} \frac{1}{n^{2}}, \  \frac{\pi^{2}}{6}\right)$




```python
x = sympy.Product(n,(n,1,7))
x, x.doit()
```




$\displaystyle \left( \prod_{n=1}^{7} n, \  5040\right)$




```python
x = sympy.Symbol("x")
```


```python
sympy.Sum((x)**n/(sympy.factorial(n)), (n, 1, oo)).doit().simplify()
```




$\displaystyle e^{x} - 1$



## Equation Solving

If an equation can be solved analytically, there is a good chance that SymPy is able to find the solution. If not, numerical methods might be the only option.

In its simplest form, equation solving involves a single equation with a single unknown variable, and no additional parameters: for example, finding the value of x that satisfy the second-degree polynomial
equation x 2 + 2 x – 3 = 0 .


```python
x = sympy.symbols("x")
sympy.solve(x**2 + 2*x - 3)
```




$\displaystyle \left[ -3, \  1\right]$



That is, the solutions are x = -3 and x = 1. 

The argument to sympy.solve is an expression that will be solved under the assumption that it equals zero. When this expression contains more than one symbol, the variable that is to be solved for must be given as a second argument.


```python
a,b,c = sympy.symbols("a, b, c")
sympy.solve(a*x**2 + b*x + c, x)
```




$\displaystyle \left[ \frac{- b - \sqrt{- 4 a c + b^{2}}}{2 a}, \  \frac{- b + \sqrt{- 4 a c + b^{2}}}{2 a}\right]$



sympy.solve can also solve trigonometric expressions and other special functions.


```python
sympy.solve(sympy.sin(x) - sympy.cos(x), x)
```




$\displaystyle \left[ \frac{\pi}{4}\right]$




```python
sympy.solve(sympy.exp(x) + 2*x, x)
```




$\displaystyle \left[ - W\left(\frac{1}{2}\right)\right]$



It is not uncommon to encounter equations that are not solvable algebraically, or that SymPy is unable to solve. 

In these cases SymPy will return a formal solution, which can be evaluated numerically, or raise an error if no method is available.


```python
sympy.solve(x**5 - x**2 + 1, x)
```




$\displaystyle \left[ \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 0\right)}, \  \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 1\right)}, \  \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 2\right)}, \  \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 3\right)}, \  \operatorname{CRootOf} {\left(x^{5} - x^{2} + 1, 4\right)}\right]$




```python
#sympy.solve(sympy.tan(x)+x, x)
# NotImplementedError: multiple generators. Debug TODO.
```

Solving equations for >1 unknown variable is a generalization of univariate solvers. Pass a list of expressions instead of a single one as first argument to sympy.solve. The second argument should be a list of symbols to solve for. 

Below: how to solve two systems that are linear and nonlinear equations in x and y, respectively.


```python
eq1 = x+2 * y-1
eq2 = x   - y+1

sympy.solve([eq1, eq2], [x, y], dict=True)
```




$\displaystyle \left[ \left\{ x : - \frac{1}{3}, \  y : \frac{2}{3}\right\}\right]$




```python
eq1 = x**2 - y
eq2 = y**2 - x

sols = sympy.solve([eq1, eq2], [x, y], dict=True)
sols
```




$\displaystyle \left[ \left\{ x : 0, \  y : 0\right\}, \  \left\{ x : 1, \  y : 1\right\}, \  \left\{ x : \left(- \frac{1}{2} - \frac{\sqrt{3} i}{2}\right)^{2}, \  y : - \frac{1}{2} - \frac{\sqrt{3} i}{2}\right\}, \  \left\{ x : \left(- \frac{1}{2} + \frac{\sqrt{3} i}{2}\right)^{2}, \  y : - \frac{1}{2} + \frac{\sqrt{3} i}{2}\right\}\right]$



sympy.solve returns a list where each element represents a solution to the equation system. 

The optional keyword argument __dict = True__ requests each solution in dictionary format, which maps the symbols that have been solved for
to their values. 

This dictionary can be used in calls to __subs__, which is used below (checks that each solution indeed satisfies the two equations).


```python
[eq1.subs(sol).simplify() == 0 and eq2.subs(sol).simplify() == 0 
 for sol in sols]
```




    [True, True, True, True]



## Linear algebra
![sympy-matrix-ops](pics/sympy-matrix-ops.png)


```python
sympy.Matrix([1,2])
```




$\displaystyle \left[\begin{matrix}1\\2\end{matrix}\right]$




```python
sympy.Matrix([[1,2]])
```




$\displaystyle \left[\begin{matrix}1 & 2\end{matrix}\right]$




```python
sympy.Matrix([[1, 2], [3, 4]])
```




$\displaystyle \left[\begin{matrix}1 & 2\\3 & 4\end{matrix}\right]$




```python
sympy.Matrix(3, 4, lambda m,n: 10 * m + n)
```




$\displaystyle \left[\begin{matrix}0 & 1 & 2 & 3\\10 & 11 & 12 & 13\\20 & 21 & 22 & 23\end{matrix}\right]$



Sympy matrix elements can be symbolic expressions.


```python
a, b, c, d = sympy.symbols("a, b, c, d")
```


```python
M = sympy.Matrix([[a, b], [c, d]])
M, M*M
```




$\displaystyle \left( \left[\begin{matrix}a & b\\c & d\end{matrix}\right], \  \left[\begin{matrix}a^{2} + b c & a b + b d\\a c + c d & b c + d^{2}\end{matrix}\right]\right)$




```python
x = sympy.Matrix(sympy.symbols("x_1, x_2"))
M*x
```




$\displaystyle \left[\begin{matrix}a x_{1} + b x_{2}\\c x_{1} + d x_{2}\end{matrix}\right]$



With purely numerical methods, we would have to choose particular values of p and q before beginning to solve this problem, for example, using an LU factorization (or by computing the inverse) of the matrix on the left-hand side of the equation. 

With a symbolic computing approach we can proceed with computing the solution, as if we did the calculation analytically by hand. With SymPy, we can simply define symbols for the unknown variables and parameters,
and setup the required matrix objects.


```python
p,q = sympy.symbols("p, q")
M = sympy.Matrix([[1,p],[q,1]])
M
```




$\displaystyle \left[\begin{matrix}1 & p\\q & 1\end{matrix}\right]$




```python
b = sympy.Matrix(sympy.symbols("b_1, b_2"))
b
```




$\displaystyle \left[\begin{matrix}b_{1}\\b_{2}\end{matrix}\right]$




```python
M.solve(b)
```




$\displaystyle \left[\begin{matrix}\frac{- b_{1} + b_{2} p}{p q - 1}\\\frac{b_{1} q - b_{2}}{p q - 1}\end{matrix}\right]$




```python
M.LUsolve(b)
```




$\displaystyle \left[\begin{matrix}b_{1} - \frac{p \left(- b_{1} q + b_{2}\right)}{- p q + 1}\\\frac{- b_{1} q + b_{2}}{- p q + 1}\end{matrix}\right]$




```python
M.inv()*b # multiple inverse of M by vector b:
```




$\displaystyle \left[\begin{matrix}\frac{b_{1}}{- p q + 1} - \frac{b_{2} p}{- p q + 1}\\- \frac{b_{1} q}{- p q + 1} + \frac{b_{2}}{- p q + 1}\end{matrix}\right]$




