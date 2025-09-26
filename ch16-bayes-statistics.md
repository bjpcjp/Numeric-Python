# Bayesian statistics
* Bayes stats: __probabilities = degree of belief__ (not observations).
* Initial system knowledge: "__prior probability__" distribution
* Updated system knowledge: "__posterior probability__" distribution
* Multi methods available for finding PP distribution:
    - __Markov Chain Monte Carlo__ (MCMC) = the widely used.


- __Simple refresher__
- __Model definition__
- __Sampling posterior distributions__
- __Linear regression__

* Relation between unconditional & conditional probabilities of two events A & B: *P(A|B) P(B) = P(B|A) P(A)*
* P(A|B) = conditional prob of A given B being true
* P(B|A) = conditional prob of B given A being true
* Use case: situations where we want to update P(A) based on event B.


```python
# pymc3:
# PyMC variables: distribution params (like mean, variance for normal distribution)
# can themselves be random variables.
# this allows model chaining.

import pymc3 as mc # probabilistic programming framework
```


```python
import numpy as np
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
```

### Simple example: Normal distributed random variable


```python
np.random.seed(100)
```


```python
mu    = 4.0
sigma = 2.0
model = mc.Model()
```


```python
with model:
    mc.Normal('X', mu, 1/sigma**2)
model.vars
```




    [X]




```python
start = dict(X=2)
```


```python
# mc.sample() -- implements MCMC for sampling from random vars in model
with model:
    step = mc.Metropolis()
    trace = mc.sample(10000, step=step, start=start)
```

    Multiprocess sampling (4 chains in 4 jobs)
    Metropolis: [X]
    Sampling 4 chains: 100%|██████████| 42000/42000 [00:03<00:00, 11322.81draws/s]
    The number of effective samples is smaller than 25% for some parameters.



```python
# now have 10K values from random variable.
# use get_values() to access them
# y = PDF of distribution

x = np.linspace(-4, 12, 1000)
y = stats.norm(mu, sigma).pdf(x)
X = trace.get_values("X")
```


```python
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x, y, 'r', lw=2)
sns.distplot(X, ax=ax)
ax.set_xlim(-4, 12)
ax.set_xlabel("x")
ax.set_ylabel("Probability distribution")
fig.tight_layout()
#fig.savefig("ch16-normal-distribution-sampled.pdf")
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_12_1.png)
    



```python
# viz MCMC random walk == source data
# mc.traceplot == kernel density plot & sampling trace (automatic)

fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), squeeze=False)
mc.traceplot(trace, ax=axes)
axes[0,0].plot(x, y, 'r', lw=0.5)
fig.tight_layout()
#fig.savefig("ch16-normal-sampling-trace.png")
#fig.savefig("ch16-normal-sampling-trace.pdf")

# below left: density kernel estimate (blue) vs sampling trace (red)
# below right: MCM sampling trace
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_13_0.png)
    


### Dependent random variables


```python
model = mc.Model()
```


```python
# normal distr, but both mean & sigma are themselves normal distribs
with model:
    mean = mc.Normal('mean', 3.0)
    sigma = mc.HalfNormal('sigma', sd=1.0)
    X = mc.Normal('X', mean, sd=sigma)
```


```python
model.vars
```




    [mean, sigma_log__, X]




```python
# not so easy to find good starting point for sampling
# use find_MAP() instead -- corresponds to maxx posterior distribution
with model:
    #start = mc.find_MAP() - userWarning: find_MAP shouldn't initialize. use sample() instead.
    start = mc.sample()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [X, sigma, mean]
    Sampling 4 chains: 100%|██████████| 4000/4000 [00:01<00:00, 2216.51draws/s]
    There were 43 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.6870625338867792, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 22 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 132 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.5163646137843638, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 251 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.25229786603481535, but should be close to 0.8. Try to increase the number of tuning steps.
    The gelman-rubin statistic is larger than 1.05 for some parameters. This indicates slight problems during sampling.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
start
```




    <MultiTrace: 4 chains, 500 iterations, 4 variables>




```python
# now we can start sampling
# using mc.Metropolis as an MCMC sampling step method
with model:
    step = mc.Metropolis()
    trace = mc.sample(100000, start=start, step=step)
```

    Multiprocess sampling (4 chains in 4 jobs)
    CompoundStep
    >Metropolis: [X]
    >Metropolis: [sigma]
    >Metropolis: [mean]



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-27-6cce057cab22> in <module>()
          3 with model:
          4     step = mc.Metropolis()
    ----> 5     trace = mc.sample(100000, start=start, step=step)
    

    ~/anaconda3/lib/python3.6/site-packages/pymc3/sampling.py in sample(draws, step, init, n_init, start, trace, chain_idx, chains, cores, tune, nuts_kwargs, step_kwargs, progressbar, model, random_seed, live_plot, discard_tuned_samples, live_plot_kwargs, compute_convergence_checks, use_mmap, **kwargs)
        437             _print_step_hierarchy(step)
        438             try:
    --> 439                 trace = _mp_sample(**sample_args)
        440             except pickle.PickleError:
        441                 _log.warning("Could not pickle model, sampling singlethreaded.")


    ~/anaconda3/lib/python3.6/site-packages/pymc3/sampling.py in _mp_sample(draws, tune, step, chains, cores, chain, random_seed, start, progressbar, trace, model, use_mmap, **kwargs)
        984         sampler = ps.ParallelSampler(
        985             draws, tune, chains, cores, random_seed, start, step,
    --> 986             chain, progressbar)
        987         try:
        988             try:


    ~/anaconda3/lib/python3.6/site-packages/pymc3/parallel_sampling.py in __init__(self, draws, tune, chains, cores, seeds, start_points, step_method, start_chain_num, progressbar)
        305 
        306         if any(len(arg) != chains for arg in [seeds, start_points]):
    --> 307             raise ValueError("Number of seeds and start_points must be %s." % chains)
        308 
        309         self._samplers = [


    ValueError: Number of seeds and start_points must be 4.



```python
# result = NumPy array with sample values
# can be used for stats
trace.get_values('sigma').mean()
```




    0.8219816833485929




```python
# same approach can be used with X
X = trace.get_values('X')
```


```python
X.mean()
```




    2.9565878858789767




```python
trace.get_values('X').std()
```




    1.4324234516333139




```python
fig, axes = plt.subplots(3, 2, figsize=(8, 6), squeeze=False)
mc.traceplot(trace, varnames=['mean', 'sigma', 'X'], ax=axes)
fig.tight_layout()
fig.savefig("ch16-dependent-rv-sample-trace.png")
fig.savefig("ch16-dependent-rv-sample-trace.pdf")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_25_0.png)
    


### Sampling Posterior distributions
* Bayes' real use case: sampling from posterior distribution == probability distrs for updated model variables.
* To condition model, use observed=data in model definition


```python
# norm distrib, mean 2.5, sigma 1.5
mu = 2.5
s = 1.5
data = stats.norm(mu, s).rvs(100)
data[0:10]
```




    array([ 0.45598376,  0.56555613,  2.49893641,  3.978023  ,  2.71354689,
            1.56389224,  2.68839463,  4.4411751 ,  3.74339254,  2.62927864])




```python
with mc.Model() as model:
    
    mean  = mc.Normal('mean', 4.0, 1.0) # true 2.5
    sigma = mc.HalfNormal('sigma', 3.0 * np.sqrt(np.pi/2)) # true 1.5

    X = mc.Normal('X', mean, 1/sigma**2, observed=data)
```


```python
# X no longer used to construct likelihoods
model.vars
```




    [mean, sigma_log_]




```python
# 
with model:
    start = mc.find_MAP()         # find appropriate start point
    step =  mc.Metropolis()       # step instance
    trace = mc.sample(100000, start=start, step=step)
    #step = mc.NUTS()
    #trace = mc.sample(10000, start=start, step=step)
```

    Optimization terminated successfully.
             Current function value: 187.304524
             Iterations: 15
             Function evaluations: 19

    100%|██████████| 100000/100000 [00:14<00:00, 6919.95it/s]

    
             Gradient evaluations: 19


    



```python
start
```




    {'mean': array(2.4049743290420094), 'sigma_log_': array(-0.20600880617494624)}




```python
model.vars
```




    [mean, sigma_log_]




```python
fig, axes = plt.subplots(2, 2, figsize=(8, 4), squeeze=False)
mc.traceplot(trace, varnames=['mean', 'sigma'], ax=axes)
fig.tight_layout()
fig.savefig("ch16-posterior-sample-trace.png")
fig.savefig("ch16-posterior-sample-trace.pdf")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_33_0.png)
    



```python
# to get stats, access arrays using get_values with var name as arg
mu, trace.get_values('mean').mean()
```




    (2.5, 2.4059325926349997)




```python
s, trace.get_values('sigma').mean()
```




    (1.5, 0.81004398217076157)




```python
# forestplot = viz of mean & credibility intervals
# for each random var in a model
gs = mc.forestplot(trace, varnames=['mean', 'sigma'])
plt.savefig("ch16-forestplot.pdf")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_36_0.png)
    



```python
mc.summary
```




    <function pymc3.stats.summary>




```python
mc.summary(trace, varnames=['mean', 'sigma'])
```

    
    mean:
    
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      2.406            0.151            0.001            [2.112, 2.708]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      2.109          2.306          2.405          2.506          2.705
    
    
    sigma:
    
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      0.810            0.029            0.000            [0.754, 0.866]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      0.753          0.790          0.810          0.830          0.866
    


### Linear regression
* use case: assigning prior probabilities to unknown slopes & intercepts.


```python
# height & weight for 200 men & women
dataset = sm.datasets.get_rdataset("Davis", "car")
```


```python
# use only males for now
# use only males < 100kg for now
data = dataset.data[dataset.data.sex == 'M']
data = data[data.weight < 110]
data.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>weight</th>
      <th>height</th>
      <th>repwt</th>
      <th>repht</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>77</td>
      <td>182</td>
      <td>77.0</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>68</td>
      <td>177</td>
      <td>70.0</td>
      <td>175.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>76</td>
      <td>170</td>
      <td>76.0</td>
      <td>165.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use statsmodel library
# ordinary least squares regrsssion
# patsy formula for height vs weight
model = smf.ols("height ~ weight", data=data)
```


```python
result = model.fit()
```


```python
result.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>height</td>      <th>  R-squared:         </th> <td>   0.327</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.319</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   41.35</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 10 May 2017</td> <th>  Prob (F-statistic):</th> <td>7.11e-09</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:05:50</td>     <th>  Log-Likelihood:    </th> <td> -268.20</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    87</td>      <th>  AIC:               </th> <td>   540.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    85</td>      <th>  BIC:               </th> <td>   545.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>  152.6173</td> <td>    3.987</td> <td>   38.281</td> <td> 0.000</td> <td>  144.691</td> <td>  160.544</td>
</tr>
<tr>
  <th>weight</th>    <td>    0.3365</td> <td>    0.052</td> <td>    6.431</td> <td> 0.000</td> <td>    0.232</td> <td>    0.441</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.734</td> <th>  Durbin-Watson:     </th> <td>   2.039</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.057</td> <th>  Jarque-Bera (JB):  </th> <td>   5.660</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.397</td> <th>  Prob(JB):          </th> <td>  0.0590</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.965</td> <th>  Cond. No.          </th> <td>    531.</td>
</tr>
</table>




```python
# model in place. let's use predict() method to try it out.
x = np.linspace(50, 110, 25)
y = result.predict({"weight": x})
```


```python
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(data.weight, data.height, 'o')
ax.plot(x, y, color="blue")
ax.set_xlabel("weight")
ax.set_ylabel("height")
fig.tight_layout()
fig.savefig("ch16-linear-ols-fit.pdf")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_46_0.png)
    



```python
# creditbility interval = describes uncertainty in estimate

with mc.Model() as model:
    sigma     = mc.Uniform('sigma', 0, 10)
    intercept = mc.Normal('intercept', 125, sd=30)
    beta      = mc.Normal('beta', 0, sd=5)
    
    height_mu = intercept + beta * data.weight

    # likelihood function
    mc.Normal('height', mu=height_mu, sd=sigma, observed=data.height)

    # predict
    predict_height = mc.Normal('predict_height', mu=intercept + beta * x, sd=sigma, shape=len(x)) 
```


```python
model.vars
```




    [sigma_interval_, intercept, beta, predict_height]




```python
with model:
    start = mc.find_MAP()
    step = mc.NUTS(state=start)
    trace = mc.sample(10000, step, start=start)
```

    Optimization terminated successfully.
             Current function value: 339.789864
             Iterations: 54
             Function evaluations: 74
             Gradient evaluations: 74

    100%|██████████| 10000/10000 [00:19<00:00, 510.42it/s]

    


    



```python
model.vars
```




    [sigma_interval_, intercept, beta, predict_height]




```python
fig, axes = plt.subplots(2, 2, figsize=(8, 4), squeeze=False)
mc.traceplot(trace, varnames=['intercept', 'beta'], ax=axes)
fig.savefig("ch16-linear-model-sample-trace.pdf")
fig.savefig("ch16-linear-model-sample-trace.png")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_51_0.png)
    



```python
intercept = trace.get_values("intercept").mean()
beta      = trace.get_values("beta").mean()
```


```python
intercept, beta
```




    (152.3558161144345, 0.33981749294464142)




```python
result.params
```




    Intercept    152.617348
    weight         0.336477
    dtype: float64




```python
#result.predict({"weight": 90})
```


```python
weight_index = np.where(x == 90)[0][0]
```


```python
trace.get_values("predict_height")[:, weight_index].mean()
```




    182.94710921928788




```python
fig, ax = plt.subplots(figsize=(8, 3))

sns.distplot(trace.get_values("predict_height")[:, weight_index], ax=ax)
ax.set_xlim(150, 210)
ax.set_xlabel("height")
ax.set_ylabel("Probability distribution")
fig.tight_layout()
fig.savefig("ch16-linear-model-predict-cut.pdf")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_58_0.png)
    



```python
fig, ax = plt.subplots(1, 1, figsize=(8, 3))

for n in range(500, 2000, 1):
    intercept = trace.get_values("intercept")[n]
    beta      = trace.get_values("beta")[n]
    
    ax.plot(x, 
            intercept + beta * x, 
            color='red', 
            lw=0.25, 
            alpha=0.05)

intercept = trace.get_values("intercept").mean()
beta      = trace.get_values("beta").mean()

ax.plot(x, intercept + beta * x, 
        color='k', 
        label="Mean Bayesian prediction")

ax.plot(data.weight, data.height, 'o')
ax.plot(x, y, '--', color="blue", label="OLS prediction")
ax.set_xlabel("weight")
ax.set_ylabel("height")
ax.legend(loc=0)

fig.tight_layout()
fig.savefig("ch16-linear-model-fit.pdf")
fig.savefig("ch16-linear-model-fit.png")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_59_0.png)
    



```python
# for general-purpose linear models
# PyMC provides simple API to create model & stochastic vars

with mc.Model() as model:
    mc.glm.glm('height ~ weight', data)
    step = mc.NUTS()
    trace = mc.sample(2000, step)
```

    100%|██████████| 2000/2000 [00:25<00:00, 78.21it/s] 



```python
fig, axes = plt.subplots(3, 2, figsize=(8, 6), squeeze=False)

# traceplot
# sd = sigma in model definition, indicates std error of residual
# note in plot how sampling needs ~100 points before steady state

mc.traceplot(
    trace, 
    varnames=['Intercept', 'weight', 'sd'], 
    ax=axes)

fig.tight_layout()
fig.savefig("ch16-glm-sample-trace.pdf")
fig.savefig("ch16-glm-sample-trace.png")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_61_0.png)
    


### Multilevel model


```python
dataset = sm.datasets.get_rdataset("Davis", "car")
```


```python
data = dataset.data.copy()
data = data[data.weight < 110]
data["sex"] = data["sex"].apply(lambda x: 1 if x == "F" else 0)
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>weight</th>
      <th>height</th>
      <th>repwt</th>
      <th>repht</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>77</td>
      <td>182</td>
      <td>77.0</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>58</td>
      <td>161</td>
      <td>51.0</td>
      <td>159.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>53</td>
      <td>161</td>
      <td>54.0</td>
      <td>158.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>68</td>
      <td>177</td>
      <td>70.0</td>
      <td>175.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>59</td>
      <td>157</td>
      <td>59.0</td>
      <td>155.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
with mc.Model() as model:

    # heirarchical model: hyper priors
    #intercept_mu = mc.Normal("intercept_mu", 125)
    #intercept_sigma = 30.0 #mc.Uniform('intercept_sigma', lower=0, upper=50)
    #beta_mu = mc.Normal("beta_mu", 0.0)
    #beta_sigma = 5.0 #mc.Uniform('beta_sigma', lower=0, upper=10)
    
    # multilevel model: prior parameters
    intercept_mu, intercept_sigma = 125, 30
    beta_mu, beta_sigma = 0.0, 5.0
    
    # priors
    intercept = mc.Normal('intercept', intercept_mu, sd=intercept_sigma, shape=2)
    beta = mc.Normal('beta', beta_mu, sd=beta_sigma, shape=2)
    error = mc.Uniform('error', 0, 10)

    # model equation
    sex_idx = data.sex.values
    height_mu = intercept[sex_idx] + beta[sex_idx] * data.weight

    mc.Normal('height', mu=height_mu, sd=error, observed=data.height)
```


```python
model.vars
```




    [intercept, beta, error_interval_]




```python
# invoke MCMC (NUTS) sampler - collect 5K samples
with model:
    start = mc.find_MAP()
    step = mc.NUTS(state=start)
    hessian = mc.find_hessian(start)
    trace = mc.sample(5000, step, start=start)
```

    Warning: Desired error not necessarily achieved due to precision loss.
             Current function value: 617.047108
             Iterations: 26
             Function evaluations: 105

    100%|██████████| 5000/5000 [00:39<00:00, 125.01it/s]

    
             Gradient evaluations: 93


    



```python
fig, axes = plt.subplots(3, 2, figsize=(8, 6), squeeze=False)

# again use traceplot. blue = male, green = female

mc.traceplot(
    trace, 
    varnames=['intercept', 'beta', 'error'], 
    ax=axes)

fig.tight_layout()
fig.savefig("ch16-multilevel-sample-trace.pdf")
fig.savefig("ch16-multilevel-sample-trace.png")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_68_0.png)
    



```python
intercept_m, intercept_f = trace.get_values('intercept').mean(axis=0)
```


```python
intercept = trace.get_values('intercept').mean()
intercept
```




    146.33486241015748




```python
beta_m, beta_f = trace.get_values('beta').mean(axis=0)
beta_m, beta_f
```




    (0.34172809278974009, 0.42647881699252721)




```python
beta = trace.get_values('beta').mean()
beta
```




    0.38410345489113301




```python
fig, ax = plt.subplots(1, 1, figsize=(8, 3))

mask_m = data.sex == 0
mask_f = data.sex == 1

ax.plot(data.weight[mask_m], data.height[mask_m], 'o', color="steelblue", label="male", alpha=0.5)
ax.plot(data.weight[mask_f], data.height[mask_f], 'o', color="green", label="female", alpha=0.5)

x = np.linspace(35, 110, 50)
ax.plot(x, intercept_m + x * beta_m, color="steelblue", label="model male group")
ax.plot(x, intercept_f + x * beta_f, color="green", label="model female group")
ax.plot(x, intercept + x * beta, color="black", label="model both groups")

ax.set_xlabel("weight")
ax.set_ylabel("height")
ax.legend(loc=0)
fig.tight_layout()
fig.savefig("ch16-multilevel-linear-model-fit.pdf")
fig.savefig("ch16-multilevel-linear-model-fit.png")
```


    
![png](ch16-bayes-statistics_files/ch16-bayes-statistics_73_0.png)
    

