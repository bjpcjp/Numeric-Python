# Statistical modeling with statsmodel & patsy

- __Intro__
- __Intro to Patsy__
- __Linear regression__
- __Example datasets__
- __Discrete regression__
    - logistic regression
    - poisson model
- __Time series__

* use case: for a set of response variables (Y), and independent variables (X), find a model that explains the relations between X & Y.


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
```


```python
import patsy
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
import numpy as np
import pandas as pd
```


```python
from scipy import stats
```


```python
import seaborn as sns
```

### Using patsy


```python
np.random.seed(123456789)
```


```python
y  = np.array([ 1,  2,  3,  4,  5])
x1 = np.array([ 6,  7,  8,  9, 10])
x2 = np.array([11, 12, 13, 14, 15])
```


```python
# linear model: Y = b0 + b1x1 + b2x2 + b3x1x2
X = np.vstack([np.ones(5), x1, x2, x1*x2]).T; X
```




    array([[  1.,   6.,  11.,  66.],
           [  1.,   7.,  12.,  84.],
           [  1.,   8.,  13., 104.],
           [  1.,   9.,  14., 126.],
           [  1.,  10.,  15., 150.]])




```python
# use NumPy least-square-fit func to find beta coefficient vector
beta, res, rank, sval = np.linalg.lstsq(X, y, rcond=None); beta
```




    array([-5.55555556e-01,  1.88888889e+00, -8.88888889e-01, -1.11022302e-15])



### Patsy syntax
* basic structure: "LHS ~ RHS" (LHS contains response vars; RHS contains independent vars)
* "+" & "-" signs in expressions are set union/difference ops, not math ops


```python
# create a dictionary to map variable names to corresponding data arrays
data = {"y": y, "x1": x1, "x2": x2}
```


```python
# define model
y, X = patsy.dmatrices("y ~ 1 + x1 + x2 + x1*x2", data)
```


```python
# y, X are DesignMatrix instances = subclass of NumPy arrays
y
```




    DesignMatrix with shape (5, 1)
      y
      1
      2
      3
      4
      5
      Terms:
        'y' (column 0)




```python
X
```




    DesignMatrix with shape (5, 4)
      Intercept  x1  x2  x1:x2
              1   6  11     66
              1   7  12     84
              1   8  13    104
              1   9  14    126
              1  10  15    150
      Terms:
        'Intercept' (column 0)
        'x1' (column 1)
        'x2' (column 2)
        'x1:x2' (column 3)




```python
np.array(X)
```




    array([[  1.,   6.,  11.,  66.],
           [  1.,   7.,  12.,  84.],
           [  1.,   8.,  13., 104.],
           [  1.,   9.,  14., 126.],
           [  1.,  10.,  15., 150.]])




```python
# ordinary linear regression (OLS)
model = sm.OLS(y, X)
```


```python
result = model.fit()
result.params
```




    array([-5.55555556e-01,  1.88888889e+00, -8.88888889e-01, -7.77156117e-16])




```python
# using statsmodel API (imported as smf)
# pass Patsy model formula & data dictionary
model = smf.ols(
    "y ~ 1 + x1 + x2 + x1:x2", 
    df_data)

result = model.fit()
result.params
```




    Intercept   -5.555556e-01
    x1           1.888889e+00
    x2          -8.888889e-01
    x1:x2       -7.771561e-16
    dtype: float64




```python
result.summary()
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/statsmodels/stats/stattools.py:72: ValueWarning: omni_normtest is not valid with less than 8 observations; 5 samples were given.
      "samples were given." % int(n), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>6.860e+27</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 19 May 2019</td> <th>  Prob (F-statistic):</th> <td>1.46e-28</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>09:19:27</td>     <th>  Log-Likelihood:    </th> <td>  151.41</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>     5</td>      <th>  AIC:               </th> <td>  -296.8</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>     2</td>      <th>  BIC:               </th> <td>  -298.0</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -0.5556</td> <td> 6.16e-14</td> <td>-9.02e+12</td> <td> 0.000</td> <td>   -0.556</td> <td>   -0.556</td>
</tr>
<tr>
  <th>x1</th>        <td>    1.8889</td> <td>  2.3e-13</td> <td> 8.22e+12</td> <td> 0.000</td> <td>    1.889</td> <td>    1.889</td>
</tr>
<tr>
  <th>x2</th>        <td>   -0.8889</td> <td> 7.82e-14</td> <td>-1.14e+13</td> <td> 0.000</td> <td>   -0.889</td> <td>   -0.889</td>
</tr>
<tr>
  <th>x1:x2</th>     <td>-7.772e-16</td> <td> 7.22e-15</td> <td>   -0.108</td> <td> 0.924</td> <td>-3.18e-14</td> <td> 3.03e-14</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>   nan</td> <th>  Durbin-Watson:     </th> <td>   0.071</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td>   nan</td> <th>  Jarque-Bera (JB):  </th> <td>   0.399</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.401</td> <th>  Prob(JB):          </th> <td>   0.819</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.871</td> <th>  Cond. No.          </th> <td>6.86e+17</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.31e-31. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
from collections import defaultdict
```


```python
data = defaultdict(lambda: np.array([1,2,3]))
```


```python
# intercept & a correspond to a constant and a linear dependence on a.
patsy.dmatrices("y ~ a", data=data)[1].design_info.term_names
```




    ['Intercept', 'a']




```python
# now we have 2nd independent variable, "b"
patsy.dmatrices("y ~ 1 + a + b", data=data)[1].design_info.term_names
```




    ['Intercept', 'a', 'b']




```python
# intercept can be removed
patsy.dmatrices("y ~ -1 + a + b", data=data)[1].design_info.term_names
```




    ['a', 'b']




```python
# auto expansion
patsy.dmatrices("y ~ a * b", data=data)[1].design_info.term_names
```




    ['Intercept', 'a', 'b', 'a:b']




```python
# higher-order expansions
patsy.dmatrices("y ~ a * b * c", data=data)[1].design_info.term_names
```




    ['Intercept', 'a', 'b', 'a:b', 'c', 'a:c', 'b:c', 'a:b:c']




```python
# removing specific term (in this case, a:b:c)
patsy.dmatrices("y ~ a * b * c - a:b:c", data=data)[1].design_info.term_names
```




    ['Intercept', 'a', 'b', 'a:b', 'c', 'a:c', 'b:c']



* "+" and "-" operators are used in Patsy for set-like operations (not math)
* Patsy also provides an identity function (I).


```python
data = {k: np.array([]) for k in ["y", "a", "b", "c"]}
```


```python
patsy.dmatrices("y ~ a + b", data=data)[1].design_info.term_names
```




    ['Intercept', 'a', 'b']




```python
patsy.dmatrices("y ~ I(a + b)", data=data)[1].design_info.term_names
```




    ['Intercept', 'I(a + b)']




```python
patsy.dmatrices("y ~ a*a", data=data)[1].design_info.term_names
```




    ['Intercept', 'a']




```python
patsy.dmatrices("y ~ I(a**2)", data=data)[1].design_info.term_names
```




    ['Intercept', 'I(a ** 2)']




```python
patsy.dmatrices("y ~ np.log(a) + b", data=data)[1].design_info.term_names
```




    ['Intercept', 'np.log(a)', 'b']




```python
z = lambda x1, x2: x1+x2
```


```python
patsy.dmatrices("y ~ z(a, b)", data=data)[1].design_info.term_names
```




    ['Intercept', 'z(a, b)']



### Categorical variables
* When using categories in a linear model, we typically need to recode them with dummy variables.


```python
data = {"y": [1, 2, 3], "a": [1, 2, 3]}
```


```python
patsy.dmatrices("y ~ - 1 + a", data=data, return_type="dataframe")[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
patsy.dmatrices("y ~ - 1 + C(a)", data=data, return_type="dataframe")[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C(a)[1]</th>
      <th>C(a)[2]</th>
      <th>C(a)[3]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# variables with non-numerical values are auto-interpreted & treated as category data.
data = {"y": [1, 2, 3], "a": ["type A", "type B", "type C"]}
```


```python
patsy.dmatrices("y ~ - 1 + a", data=data, return_type="dataframe")[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a[type A]</th>
      <th>a[type B]</th>
      <th>a[type C]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Category data encoding is binary by default.
# Encoding can be changed/extended.
# Example: encoding categories with orthogonal polynomials instead of treatment indicators
# using (C(a,Poly))

patsy.dmatrices("y ~ - 1 + C(a, Poly)", data=data, return_type="dataframe")[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C(a, Poly).Constant</th>
      <th>C(a, Poly).Linear</th>
      <th>C(a, Poly).Quadratic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>-7.071068e-01</td>
      <td>0.408248</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-5.551115e-17</td>
      <td>-0.816497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>7.071068e-01</td>
      <td>0.408248</td>
    </tr>
  </tbody>
</table>
</div>



### Example: Linear regression
* Basic workflow:
    - Create instance of model class
    - invoke fit method
    - print summary stats
    - post-process model fit results


```python
np.random.seed(123456789)
N = 100
x1 = np.random.randn(N)
x2 = np.random.randn(N)
```


```python
data = pd.DataFrame({"x1": x1, "x2": x2})
```


```python
# true value: y = 1 + 2x1 + 3x2 + 4x12

def y_true(x1, x2):
    return 1 + 2*x1 + 3*x2 + 4*x1*x2
```


```python
# store true value of y in y_true column of the DataFrame.
data["y_true"] = y_true(x1, x2)
data["y_true"][0:5]
```




    0    -0.198823
    1   -12.298805
    2   -15.420705
    3     2.313945
    4    -1.282107
    Name: y_true, dtype: float64




```python
# add normal-distributed noise to true values
e = np.random.randn(N)
e[0:5]
```




    array([-0.56463281,  0.56190437,  1.22155556, -1.08257084, -0.35554624])




```python
data["y"] = data["y_true"] + e
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>y_true</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.212902</td>
      <td>-0.474588</td>
      <td>-0.198823</td>
      <td>-0.763456</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.128398</td>
      <td>-1.524772</td>
      <td>-12.298805</td>
      <td>-11.736900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.841711</td>
      <td>-1.939271</td>
      <td>-15.420705</td>
      <td>-14.199150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.082382</td>
      <td>0.345148</td>
      <td>2.313945</td>
      <td>1.231374</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.858964</td>
      <td>-0.621523</td>
      <td>-1.282107</td>
      <td>-1.637653</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we have two explanatory values (x1,x2) and a response (y).
# start with linear model
model = smf.ols("y ~ x1 + x2", data)
result = model.fit()
```


```python
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.376
    Model:                            OLS   Adj. R-squared:                  0.363
    Method:                 Least Squares   F-statistic:                     29.19
    Date:                Sun, 19 May 2019   Prob (F-statistic):           1.19e-10
    Time:                        09:41:05   Log-Likelihood:                -269.97
    No. Observations:                 100   AIC:                             545.9
    Df Residuals:                      97   BIC:                             553.8
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      1.0444      0.377      2.774      0.007       0.297       1.792
    x1             1.1313      0.385      2.940      0.004       0.368       1.895
    x2             2.9668      0.425      6.981      0.000       2.123       3.810
    ==============================================================================
    Omnibus:                       16.990   Durbin-Watson:                   1.619
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.548
    Skew:                          -0.640   Prob(JB):                     5.19e-08
    Kurtosis:                       5.533   Cond. No.                         1.32
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# r-squared: how does data fit model? (1.0 = perfect)
result.rsquared
```




    0.3757287272091422




```python
# to investigate whether assumption of normal-distributed errors is justified
# start with residuals
result.resid.head()
```




    0    -2.903191
    1   -10.665331
    2   -11.573520
    3    -0.930212
    4    -1.809809
    dtype: float64




```python
# check for normality
z, p = stats.normaltest(result.fittedvalues.values)
p
```




    0.690964627386556




```python
# extract coefficients
result.params
```




    Intercept    1.044399
    x1           1.131253
    x2           2.966821
    dtype: float64




```python
# QQ plot - compares sample quantiles with theoretical quantiles
# should be similar to a straight line if sampled values are normally distributed.
fig, ax = plt.subplots(figsize=(8, 4))
smg.qqplot(result.resid, ax=ax)

fig.tight_layout()
#fig.savefig("ch14-qqplot-model-1.pdf")
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_59_0.png)
    


* Above significant deviation == unlikely to be a sample from normal distribution. We need to refine the model.
* Can add missing interaction to Patsy formula, then repeat.


```python
model = smf.ols("y ~ x1 + x2 + x1*x2", data)
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.950
    Model:                            OLS   Adj. R-squared:                  0.949
    Method:                 Least Squares   F-statistic:                     614.3
    Date:                Sun, 19 May 2019   Prob (F-statistic):           1.70e-62
    Time:                        09:46:58   Log-Likelihood:                -143.25
    No. Observations:                 100   AIC:                             294.5
    Df Residuals:                      96   BIC:                             304.9
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.9304      0.107      8.725      0.000       0.719       1.142
    x1             2.0025      0.112     17.877      0.000       1.780       2.225
    x2             2.8567      0.120     23.738      0.000       2.618       3.096
    x1:x2          3.8681      0.116     33.384      0.000       3.638       4.098
    ==============================================================================
    Omnibus:                        2.046   Durbin-Watson:                   1.814
    Prob(Omnibus):                  0.359   Jarque-Bera (JB):                1.774
    Skew:                          -0.326   Prob(JB):                        0.412
    Kurtosis:                       3.011   Cond. No.                         1.38
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# r-squared stat: much better.
result.rsquared
```




    0.9504900208668222




```python
# repeat the QQ plot. (It's better.)
fig, ax = plt.subplots(figsize=(8, 4))
smg.qqplot(result.resid, ax=ax)
fig.tight_layout()
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_63_0.png)
    



```python
result.params
```




    Intercept    0.930429
    x1           2.002468
    x2           2.856701
    x1:x2        3.868131
    dtype: float64



* Predict values of new observations using __predict__.


```python
x = np.linspace(-1, 1, 50)
X1, X2 = np.meshgrid(x, x)
```


```python
new_data = pd.DataFrame(
    {"x1": X1.ravel(), 
     "x2": X2.ravel()})
```


```python
# predict y values
y_pred = result.predict(new_data)
y_pred.shape
```




    (2500,)




```python
# resize to square matrix for plotting purposes
y_pred = y_pred.values.reshape(50, 50)
```


```python
# do contour plots - true model vs fitted (100 noisy obs) model

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

def plot_y_contour(ax, Y, title):
    c = ax.contourf(X1, X2, Y, 15, cmap=plt.cm.RdBu)
    ax.set_xlabel(r"$x_1$", fontsize=20)
    ax.set_ylabel(r"$x_2$", fontsize=20)
    ax.set_title(title)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"$y$", fontsize=20)

plot_y_contour(axes[0], y_true(X1, X2), "true relation")
plot_y_contour(axes[1], y_pred, "fitted model")

fig.tight_layout()
#fig.savefig("ch14-comparison-model-true.pdf")
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_70_0.png)
    


### Datasets from R

* Datasets sourced from http://vincentarelbundock.github.io/Rdatasets/datasets.html


```python
dataset = sm.datasets.get_rdataset("Icecream", "Ecdat")
dataset.title
```




    'Ice Cream Consumption'




```python
dataset.data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30 entries, 0 to 29
    Data columns (total 4 columns):
    cons      30 non-null float64
    income    30 non-null int64
    price     30 non-null float64
    temp      30 non-null int64
    dtypes: float64(2), int64(2)
    memory usage: 1.0 KB



```python
# ordinary least squares regression
model = smf.ols(
    "cons ~ -1 + price + temp", 
    data=dataset.data)
result = model.fit()
```


```python
result.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>cons</td>       <th>  R-squared:         </th> <td>   0.986</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.985</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1001.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 19 May 2019</td> <th>  Prob (F-statistic):</th> <td>9.03e-27</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:52:26</td>     <th>  Log-Likelihood:    </th> <td>  51.903</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    30</td>      <th>  AIC:               </th> <td>  -99.81</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    28</td>      <th>  BIC:               </th> <td>  -97.00</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>price</th> <td>    0.7254</td> <td>    0.093</td> <td>    7.805</td> <td> 0.000</td> <td>    0.535</td> <td>    0.916</td>
</tr>
<tr>
  <th>temp</th>  <td>    0.0032</td> <td>    0.000</td> <td>    6.549</td> <td> 0.000</td> <td>    0.002</td> <td>    0.004</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.350</td> <th>  Durbin-Watson:     </th> <td>   0.637</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.069</td> <th>  Jarque-Bera (JB):  </th> <td>   3.675</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.776</td> <th>  Prob(JB):          </th> <td>   0.159</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.729</td> <th>  Cond. No.          </th> <td>    593.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# maybe ice cream consumption = linear corr to temp, no relation to price?

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

smg.plot_fit(result, 0, ax=ax1)
smg.plot_fit(result, 1, ax=ax2)

fig.tight_layout()
#fig.savefig("ch14-regressionplots.pdf")
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_76_0.png)
    



```python
# sure looks that way

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.regplot("price", "cons", dataset.data, ax=ax1);
sns.regplot("temp", "cons", dataset.data, ax=ax2);

fig.tight_layout()
#fig.savefig("ch14-regressionplots-seaborn.pdf")
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_77_1.png)
    


### Discrete regression, logistic regression
* Requires different techniques, because linear regression requires a normally distributed continuous variable.
* statmodel discrete regression support: **Logit** (logistic regression), **Probit** (uses CMF of normal distribution, transforms linear predictor to [0,1], **MNLogit** (multinomial logistic regression), and **Poisson** classes.


```python
# Iris dataset
df = sm.datasets.get_rdataset("iris").data
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
    Sepal.Length    150 non-null float64
    Sepal.Width     150 non-null float64
    Petal.Length    150 non-null float64
    Petal.Width     150 non-null float64
    Species         150 non-null object
    dtypes: float64(4), object(1)
    memory usage: 5.9+ KB



```python
# 3 unique species
df.Species.unique()
```




    array(['setosa', 'versicolor', 'virginica'], dtype=object)




```python
# let's use versicolor & virginica species as basis for binary variable
df_subset = df[df.Species.isin(["versicolor", "virginica"])].copy()
```


```python
# map two species names into binary
df_subset.Species = df_subset.Species.map(
    {"versicolor": 1, 
     "virginica": 0})
```


```python
# clean up names so Python doesn't have problems (periods to underscores)
df_subset.rename(
    columns={
        "Sepal.Length": "Sepal_Length", 
        "Sepal.Width": "Sepal_Width",
        "Petal.Length": "Petal_Length", 
        "Petal.Width": "Petal_Width"}, inplace=True)
```


```python
df_subset.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal_Length</th>
      <th>Sepal_Width</th>
      <th>Petal_Length</th>
      <th>Petal_Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>7.0</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>6.4</td>
      <td>3.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>6.9</td>
      <td>3.1</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use Patsy to create a logit model 
model = smf.logit(
    "Species ~ Sepal_Length + Sepal_Width + Petal_Length + Petal_Width", 
    data=df_subset)
```


```python
result = model.fit()
```

    Optimization terminated successfully.
             Current function value: 0.059493
             Iterations 12



```python
print(result.summary())
```

                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                Species   No. Observations:                  100
    Model:                          Logit   Df Residuals:                       95
    Method:                           MLE   Df Model:                            4
    Date:                Sun, 19 May 2019   Pseudo R-squ.:                  0.9142
    Time:                        10:06:06   Log-Likelihood:                -5.9493
    converged:                       True   LL-Null:                       -69.315
                                            LLR p-value:                 1.947e-26
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept       42.6378     25.708      1.659      0.097      -7.748      93.024
    Sepal_Length     2.4652      2.394      1.030      0.303      -2.228       7.158
    Sepal_Width      6.6809      4.480      1.491      0.136      -2.099      15.461
    Petal_Length    -9.4294      4.737     -1.990      0.047     -18.714      -0.145
    Petal_Width    -18.2861      9.743     -1.877      0.061     -37.381       0.809
    ================================================================================
    
    Possibly complete quasi-separation: A fraction 0.60 of observations can be
    perfectly predicted. This might indicate that there is complete
    quasi-separation. In this case some parameters will not be identified.



```python
# get_margeff = returns info on marginal effects of each explanatory variable.
print(result.get_margeff().summary())
```

            Logit Marginal Effects       
    =====================================
    Dep. Variable:                Species
    Method:                          dydx
    At:                           overall
    ================================================================================
                      dy/dx    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Sepal_Length     0.0445      0.038      1.163      0.245      -0.031       0.120
    Sepal_Width      0.1207      0.064      1.891      0.059      -0.004       0.246
    Petal_Length    -0.1703      0.057     -2.965      0.003      -0.283      -0.058
    Petal_Width     -0.3303      0.110     -2.998      0.003      -0.546      -0.114
    ================================================================================



```python
# use fitted model to predict responses to new explanatory variable values.
# TODO
```

### Poisson distribution
* use case: response variable = #successes for many attempts - each with low probability of success.


```python
# discoveries dataset
dataset = sm.datasets.get_rdataset("discoveries")
```


```python
#dataset.data.head

df = dataset.data.set_index("time").rename(columns={"value" : "discoveries"})
df.head(10).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>time</th>
      <th>1860</th>
      <th>1861</th>
      <th>1862</th>
      <th>1863</th>
      <th>1864</th>
      <th>1865</th>
      <th>1866</th>
      <th>1867</th>
      <th>1868</th>
      <th>1869</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>discoveries</th>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(1, 1, figsize=(16, 4))
df.plot(kind='bar', ax=ax)
fig.tight_layout()
#fig.savefig("ch14-discoveries.pdf")
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_93_0.png)
    



```python
# attempt to fit to a poisson process
# patsy formula "discoveries ~ 1" == model discoveries variable with only intercept coeff.

model = smf.poisson("discoveries ~ 1", data=df)
result = model.fit()
result.summary()
```

    Optimization terminated successfully.
             Current function value: 2.168457
             Iterations 1





<table class="simpletable">
<caption>Poisson Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>discoveries</td>   <th>  No. Observations:  </th>  <td>   100</td> 
</tr>
<tr>
  <th>Model:</th>              <td>Poisson</td>     <th>  Df Residuals:      </th>  <td>    99</td> 
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     0</td> 
</tr>
<tr>
  <th>Date:</th>          <td>Sun, 19 May 2019</td> <th>  Pseudo R-squ.:     </th>  <td> 0.000</td> 
</tr>
<tr>
  <th>Time:</th>              <td>10:19:48</td>     <th>  Log-Likelihood:    </th> <td> -216.85</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -216.85</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>  <td>   nan</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    1.1314</td> <td>    0.057</td> <td>   19.920</td> <td> 0.000</td> <td>    1.020</td> <td>    1.243</td>
</tr>
</table>




```python
# lambda param of poisson distribution via exponential function
# use to compare histogram of observed counts vs theoretical results
lmbda = np.exp(result.params) 
```


```python
X = stats.poisson(lmbda)
```


```python
# confidence intervals
result.conf_int()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>1.020084</td>
      <td>1.242721</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create upper & lower confidence interval bounds
X_ci_l = stats.poisson(np.exp(result.conf_int().values)[0, 0])
X_ci_u = stats.poisson(np.exp(result.conf_int().values)[0, 1])
```


```python
v, k = np.histogram(df.values, bins=12, range=(0, 12), normed=True)
#v, k = np.histogram(df.values, bins=12, range=(0, 12))
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: VisibleDeprecationWarning: Passing `normed=True` on non-uniform bins has always been broken, and computes neither the probability density function nor the probability mass function. The result is only correct if the bins are uniform, when density=True will produce the same result anyway. The argument will be removed in a future version of numpy.
      """Entry point for launching an IPython kernel.



```python
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.bar(
    k[:-1], v, 
    color="steelblue",  align='center', 
    label='Dicoveries per year') 

ax.bar(
    k-0.125, X_ci_l.pmf(k), 
    color="red", alpha=0.5, align='center', width=0.25, 
    label='Poisson fit (CI, lower)')

ax.bar(
    k, X.pmf(k), 
    color="green",  align='center', width=0.5, 
    label='Poisson fit')

ax.bar(
    k+0.125, X_ci_u.pmf(k), 
    color="red",  alpha=0.5, align='center', width=0.25, 
    label='Poisson fit (CI, upper)')

ax.legend()
fig.tight_layout()
#fig.savefig("ch14-discoveries-per-year.pdf")

# conclusion:
# dataset NOT well described by poisson process
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_100_0.png)
    


### Time series
* Not same as regular regression - time series samples can't be regarded as independent random samples.
* example model type for time series = autoregressive (AR) model - future value depends on "p" earlier values. AR = special case of ARMA (autoregressive with moving average) model.


```python
# outdoor temp dataset
df = pd.read_csv(
    "temperature_outdoor_2014.tsv", 
    header=None, 
    delimiter="\t", 
    names=["time", "temp"])

df.time = pd.to_datetime(df.time, unit="s")
df      = df.set_index("time").resample("H").mean()
```


```python
# extract March & April data to new dataframes
df_march = df[df.index.month == 3]
df_april = df[df.index.month == 4]
```

* Attempt to model temp observations using AR model.
* Important assumption: applied to "stationary process" (no autocorrelation or other trends other than those explained by model terms)


```python
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

smg.tsa.plot_acf(df_march.temp,                               lags=72, ax=axes[0])
smg.tsa.plot_acf(df_march.temp.diff().dropna(),               lags=72, ax=axes[1])
smg.tsa.plot_acf(df_march.temp.diff().diff().dropna(),        lags=72, ax=axes[2])
smg.tsa.plot_acf(df_march.temp.diff().diff().diff().dropna(), lags=72, ax=axes[3])

fig.tight_layout()
#fig.savefig("ch14-timeseries-autocorrelation.pdf")

# below: clear correlation between successive values in left-most time series
# also: decreasing correlation for increasing order.
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_105_0.png)
    



```python
# create AR model using March data
model = sm.tsa.AR(df_march.temp)
```


```python
# set fit order to 72 hours
result = model.fit(72)
```


```python
# Durbin-Watson statistical test - for stationary behavior in a time series
sm.stats.durbin_watson(result.resid)
```




    1.9985623006352973




```python
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
smg.tsa.plot_acf(result.resid, lags=72, ax=ax)
fig.tight_layout()
#fig.savefig("ch14-timeseries-resid-acf.pdf")
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_109_0.png)
    



```python
# plot forecast (red) vs prev 3 days actual (blue) vs actual outcome (green)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax.plot(
    df_march.index.values[-72:], 
    df_march.temp.values[-72:], 
    label="train data")

ax.plot(
    df_april.index.values[:72], 
    df_april.temp.values[:72], 
    label="actual outcome")

ax.plot(
    pd.date_range("2014-04-01", "2014-04-4", freq="H").values,
    result.predict("2014-04-01", "2014-04-4"), 
    label="predicted outcome")

ax.legend()
fig.tight_layout()
#fig.savefig("ch14-timeseries-prediction.pdf")
```


    
![png](ch14-statistical-modeling_files/ch14-statistical-modeling_110_0.png)
    

