# Data Processing with Pandas and Seaborn
- __Series__
- __DataFrame__
- __Time Series__
- __Intro to Seaborn__


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
import numpy as np
import pandas as pd
#pd.set_option('display.mpl_style', 'default')
plt.style.use('default')
```

## Series object

* Feature: ability to index data with labels instead of integers.


```python
s = pd.Series([909976, 8615246, 2872086, 2273305])
s
```




    0     909976
    1    8615246
    2    2872086
    3    2273305
    dtype: int64




```python
type(s)
```




    pandas.core.series.Series




```python
s.dtype, s.index, s.values
```




    (dtype('int64'),
     RangeIndex(start=0, stop=4, step=1),
     array([ 909976, 8615246, 2872086, 2273305]))




```python
s.index = ["Stockholm", "London", "Rome", "Paris"]
s.name = "Population"
s
```




    Stockholm     909976
    London       8615246
    Rome         2872086
    Paris        2273305
    Name: Population, dtype: int64




```python
s = pd.Series([909976, 8615246, 2872086, 2273305], 
              index=["Stockholm", "London", "Rome", "Paris"], name="Population")
```


```python
# can access by index (label), or directly via attribute with same name
s["London"], s.Stockholm
```




    (8615246, 909976)




```python
s[["Paris", "Rome"]]
```




    Paris    2273305
    Rome     2872086
    Name: Population, dtype: int64




```python
# descriptive stats
s.median(), s.mean(), s.std(), s.min(), s.max()
```




    (2572695.5, 3667653.25, 3399048.5005155364, 909976, 8615246)




```python
s.quantile(q=0.25), s.quantile(q=0.5), s.quantile(q=0.75)
```




    (1932472.75, 2572695.5, 4307876.0)




```python
# attributes summary
s.describe()
```




    count    4.000000e+00
    mean     3.667653e+06
    std      3.399049e+06
    min      9.099760e+05
    25%      1.932473e+06
    50%      2.572696e+06
    75%      4.307876e+06
    max      8.615246e+06
    Name: Population, dtype: float64




```python
# visualization - line plot, bar plot, box plot, pic chart

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

s.plot(ax=axes[0], kind='line', title="line")
s.plot(ax=axes[1], kind='bar', title="bar")
s.plot(ax=axes[2], kind='box', title="box")
s.plot(ax=axes[3], kind='pie', title="pie")

fig.tight_layout()
#fig.savefig("ch12-series-plot.pdf")
#fig.savefig("ch12-series-plot.png")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_14_0.png)
    


### DataFrame object
* Can be viewed as collection of Series objects with common index


```python
df = pd.DataFrame([[909976, 8615246, 2872086, 2273305],
                   ["Sweden", "United kingdom", "Italy", "France"]])
df
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
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>909976</td>
      <td>8615246</td>
      <td>2872086</td>
      <td>2273305</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sweden</td>
      <td>United kingdom</td>
      <td>Italy</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United kingdom"], 
                   [2872086, "Italy"],
                   [2273305, "France"]])
df
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
      <th>0</th>
      <td>909976</td>
      <td>Sweden</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8615246</td>
      <td>United kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2872086</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2273305</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
# setup for labeled indexing, either columns or rows
df.index = ["Stockholm", "London", "Rome", "Paris"]
df.columns = ["Population", "State"]
df
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
      <th>Population</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Stockholm</th>
      <td>909976</td>
      <td>Sweden</td>
    </tr>
    <tr>
      <th>London</th>
      <td>8615246</td>
      <td>United kingdom</td>
    </tr>
    <tr>
      <th>Rome</th>
      <td>2872086</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>Paris</th>
      <td>2273305</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
# setup during initial creation
df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United kingdom"], 
                   [2872086, "Italy"],
                   [2273305, "France"]],
                  index=["Stockholm", "London", "Rome", "Paris"],
                  columns=["Population", "State"])
df
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
      <th>Population</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Stockholm</th>
      <td>909976</td>
      <td>Sweden</td>
    </tr>
    <tr>
      <th>London</th>
      <td>8615246</td>
      <td>United kingdom</td>
    </tr>
    <tr>
      <th>Rome</th>
      <td>2872086</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>Paris</th>
      <td>2273305</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
# alternative: use dict objects to create a dataframe

df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305],
                   "State": ["Sweden", "United kingdom", "Italy", "France"]},
                  index=["Stockholm", "London", "Rome", "Paris"])
df
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
      <th>Population</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Stockholm</th>
      <td>909976</td>
      <td>Sweden</td>
    </tr>
    <tr>
      <th>London</th>
      <td>8615246</td>
      <td>United kingdom</td>
    </tr>
    <tr>
      <th>Rome</th>
      <td>2872086</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>Paris</th>
      <td>2273305</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    Index(['Stockholm', 'London', 'Rome', 'Paris'], dtype='object')




```python
df.columns
```




    Index(['Population', 'State'], dtype='object')




```python
df.values
```




    array([[909976, 'Sweden'],
           [8615246, 'United kingdom'],
           [2872086, 'Italy'],
           [2273305, 'France']], dtype=object)




```python
df.Population
```




    Stockholm     909976
    London       8615246
    Rome         2872086
    Paris        2273305
    Name: Population, dtype: int64




```python
df["Population"]
```




    Stockholm     909976
    London       8615246
    Rome         2872086
    Paris        2273305
    Name: Population, dtype: int64




```python
# extracting a column from a dataframe returns a series object
type(df.Population)
```




    pandas.core.series.Series




```python
df.Population.Stockholm
```




    909976




```python
# access dataframe rows with the loc attribute.
type(df.loc)
```




    pandas.core.indexing._LocIndexer




```python
# extracting column from df == new Series object
df.loc["Stockholm"]
```




    Population    909976
    State         Sweden
    Name: Stockholm, dtype: object




```python
# passing list of row labels == new DataFrame
df.loc[["Paris", "Rome"]]
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
      <th>Population</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Paris</th>
      <td>2273305</td>
      <td>France</td>
    </tr>
    <tr>
      <th>Rome</th>
      <td>2872086</td>
      <td>Italy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# selecting rows & cols simultaneously == new DataFrame, Series, or element value
df.loc[["Paris", "Rome"], "Population"]
```




    Paris    2273305
    Rome     2872086
    Name: Population, dtype: int64




```python
# descriptive statistics - mean, std, median, min, max, ...
df.mean()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[30], line 2
          1 # descriptive statistics - mean, std, median, min, max, ...
    ----> 2 df.mean()


    File ~/.local/lib/python3.11/site-packages/pandas/core/frame.py:11335, in DataFrame.mean(self, axis, skipna, numeric_only, **kwargs)
      11327 @doc(make_doc("mean", ndim=2))
      11328 def mean(
      11329     self,
       (...)
      11333     **kwargs,
      11334 ):
    > 11335     result = super().mean(axis, skipna, numeric_only, **kwargs)
      11336     if isinstance(result, Series):
      11337         result = result.__finalize__(self, method="mean")


    File ~/.local/lib/python3.11/site-packages/pandas/core/generic.py:11984, in NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
      11977 def mean(
      11978     self,
      11979     axis: Axis | None = 0,
       (...)
      11982     **kwargs,
      11983 ) -> Series | float:
    > 11984     return self._stat_function(
      11985         "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
      11986     )


    File ~/.local/lib/python3.11/site-packages/pandas/core/generic.py:11941, in NDFrame._stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
      11937 nv.validate_func(name, (), kwargs)
      11939 validate_bool_kwarg(skipna, "skipna", none_allowed=False)
    > 11941 return self._reduce(
      11942     func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
      11943 )


    File ~/.local/lib/python3.11/site-packages/pandas/core/frame.py:11204, in DataFrame._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
      11200     df = df.T
      11202 # After possibly _get_data and transposing, we are now in the
      11203 #  simple case where we can use BlockManager.reduce
    > 11204 res = df._mgr.reduce(blk_func)
      11205 out = df._constructor_from_mgr(res, axes=res.axes).iloc[0]
      11206 if out_dtype is not None and out.dtype != "boolean":


    File ~/.local/lib/python3.11/site-packages/pandas/core/internals/managers.py:1459, in BlockManager.reduce(self, func)
       1457 res_blocks: list[Block] = []
       1458 for blk in self.blocks:
    -> 1459     nbs = blk.reduce(func)
       1460     res_blocks.extend(nbs)
       1462 index = Index([None])  # placeholder


    File ~/.local/lib/python3.11/site-packages/pandas/core/internals/blocks.py:377, in Block.reduce(self, func)
        371 @final
        372 def reduce(self, func) -> list[Block]:
        373     # We will apply the function and reshape the result into a single-row
        374     #  Block with the same mgr_locs; squeezing will be done at a higher level
        375     assert self.ndim == 2
    --> 377     result = func(self.values)
        379     if self.values.ndim == 1:
        380         res_values = result


    File ~/.local/lib/python3.11/site-packages/pandas/core/frame.py:11136, in DataFrame._reduce.<locals>.blk_func(values, axis)
      11134         return np.array([result])
      11135 else:
    > 11136     return op(values, axis=axis, skipna=skipna, **kwds)


    File ~/.local/lib/python3.11/site-packages/pandas/core/nanops.py:147, in bottleneck_switch.__call__.<locals>.f(values, axis, skipna, **kwds)
        145         result = alt(values, axis=axis, skipna=skipna, **kwds)
        146 else:
    --> 147     result = alt(values, axis=axis, skipna=skipna, **kwds)
        149 return result


    File ~/.local/lib/python3.11/site-packages/pandas/core/nanops.py:404, in _datetimelike_compat.<locals>.new_func(values, axis, skipna, mask, **kwargs)
        401 if datetimelike and mask is None:
        402     mask = isna(values)
    --> 404 result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        406 if datetimelike:
        407     result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)


    File ~/.local/lib/python3.11/site-packages/pandas/core/nanops.py:720, in nanmean(values, axis, skipna, mask)
        718 count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
        719 the_sum = values.sum(axis, dtype=dtype_sum)
    --> 720 the_sum = _ensure_numeric(the_sum)
        722 if axis is not None and getattr(the_sum, "ndim", False):
        723     count = cast(np.ndarray, count)


    File ~/.local/lib/python3.11/site-packages/pandas/core/nanops.py:1678, in _ensure_numeric(x)
       1675 inferred = lib.infer_dtype(x)
       1676 if inferred in ["string", "mixed"]:
       1677     # GH#44008, GH#36703 avoid casting e.g. strings to numeric
    -> 1678     raise TypeError(f"Could not convert {x} to numeric")
       1679 try:
       1680     x = x.astype(np.complex128)


    TypeError: Could not convert ['SwedenUnited kingdomItalyFrance'] to numeric



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4 entries, Stockholm to Paris
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   Population  4 non-null      int64 
     1   State       4 non-null      object
    dtypes: int64(1), object(1)
    memory usage: 268.0+ bytes



```python
# datatypes for each column
df.dtypes
```




    Population     int64
    State         object
    dtype: object



### Larger datasets
* example creates a dataframe from a CSV file.
* head(), tail() = handy methods to show truncated dataset.


```python
# use head() to review first few records.
df_pop = pd.read_csv("european_cities.csv")
df_pop.head()
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
      <th>Rank</th>
      <th>City</th>
      <th>State</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>London[2]</td>
      <td>United Kingdom</td>
      <td>8,615,246</td>
      <td>1 June 2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Berlin</td>
      <td>Germany</td>
      <td>3,437,916</td>
      <td>31 May 2014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Madrid</td>
      <td>Spain</td>
      <td>3,165,235</td>
      <td>1 January 2014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Rome</td>
      <td>Italy</td>
      <td>2,872,086</td>
      <td>30 September 2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Paris</td>
      <td>France</td>
      <td>2,273,305</td>
      <td>1 January 2013</td>
    </tr>
  </tbody>
</table>
</div>




```python
# additional args for read_csv()
df_pop = pd.read_csv("european_cities.csv", delimiter=",", encoding="utf-8", header=0)
df_pop.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 105 entries, 0 to 104
    Data columns (total 5 columns):
     #   Column                   Non-Null Count  Dtype 
    ---  ------                   --------------  ----- 
     0   Rank                     105 non-null    int64 
     1   City                     105 non-null    object
     2   State                    105 non-null    object
     3   Population               105 non-null    object
     4   Date of census/estimate  105 non-null    object
    dtypes: int64(1), object(4)
    memory usage: 4.2+ KB



```python
df_pop.head()
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
      <th>Rank</th>
      <th>City</th>
      <th>State</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>London[2]</td>
      <td>United Kingdom</td>
      <td>8,615,246</td>
      <td>1 June 2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Berlin</td>
      <td>Germany</td>
      <td>3,437,916</td>
      <td>31 May 2014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Madrid</td>
      <td>Spain</td>
      <td>3,165,235</td>
      <td>1 January 2014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Rome</td>
      <td>Italy</td>
      <td>2,872,086</td>
      <td>30 September 2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Paris</td>
      <td>France</td>
      <td>2,273,305</td>
      <td>1 January 2013</td>
    </tr>
  </tbody>
</table>
</div>




```python
# apply(): tool for transforming content in a column. returns new Series object.
# remove commas from population fields & recast as integers

df_pop["NumericPopulation"] = df_pop.Population.apply(
    lambda x: int(x.replace(",", "")))

df_pop["State"].values[:3]
```




    array([' United Kingdom', ' Germany', ' Spain'], dtype=object)




```python
# remove whitespace from state field

df_pop["State"] = df_pop["State"].apply(
    lambda x: x.strip())

df_pop.head()
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
      <th>Rank</th>
      <th>City</th>
      <th>State</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
      <th>NumericPopulation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>London[2]</td>
      <td>United Kingdom</td>
      <td>8,615,246</td>
      <td>1 June 2014</td>
      <td>8615246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Berlin</td>
      <td>Germany</td>
      <td>3,437,916</td>
      <td>31 May 2014</td>
      <td>3437916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Madrid</td>
      <td>Spain</td>
      <td>3,165,235</td>
      <td>1 January 2014</td>
      <td>3165235</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Rome</td>
      <td>Italy</td>
      <td>2,872,086</td>
      <td>30 September 2014</td>
      <td>2872086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Paris</td>
      <td>France</td>
      <td>2,273,305</td>
      <td>1 January 2013</td>
      <td>2273305</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pop.dtypes
```




    Rank                        int64
    City                       object
    State                      object
    Population                 object
    Date of census/estimate    object
    NumericPopulation           int64
    dtype: object




```python
# use city field as index, and sort
df_pop2 = df_pop.set_index("City")
df_pop2 = df_pop2.sort_index()
```


```python
df_pop2.head()
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
      <th>Rank</th>
      <th>State</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
      <th>NumericPopulation</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aarhus</th>
      <td>92</td>
      <td>Denmark</td>
      <td>326,676</td>
      <td>1 October 2014</td>
      <td>326676</td>
    </tr>
    <tr>
      <th>Alicante</th>
      <td>86</td>
      <td>Spain</td>
      <td>334,678</td>
      <td>1 January 2012</td>
      <td>334678</td>
    </tr>
    <tr>
      <th>Amsterdam</th>
      <td>23</td>
      <td>Netherlands</td>
      <td>813,562</td>
      <td>31 May 2014</td>
      <td>813562</td>
    </tr>
    <tr>
      <th>Antwerp</th>
      <td>59</td>
      <td>Belgium</td>
      <td>510,610</td>
      <td>1 January 2014</td>
      <td>510610</td>
    </tr>
    <tr>
      <th>Athens</th>
      <td>34</td>
      <td>Greece</td>
      <td>664,046</td>
      <td>24 May 2011</td>
      <td>664046</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create hierarchical index via list of column names
# sortlevel(0) = sort by state

# sortlevel deprecated
#df_pop3 = df_pop.set_index(["State", "City"]).sortlevel(0)

df_pop3 = df_pop.set_index(["State", "City"]).sort_index(level=0)
df_pop3.head(7)
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
      <th></th>
      <th>Rank</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
      <th>NumericPopulation</th>
    </tr>
    <tr>
      <th>State</th>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Austria</th>
      <th>Vienna</th>
      <td>7</td>
      <td>1,794,770</td>
      <td>1 January 2015</td>
      <td>1794770</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Belgium</th>
      <th>Antwerp</th>
      <td>59</td>
      <td>510,610</td>
      <td>1 January 2014</td>
      <td>510610</td>
    </tr>
    <tr>
      <th>Brussels[17]</th>
      <td>16</td>
      <td>1,175,831</td>
      <td>1 January 2014</td>
      <td>1175831</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Bulgaria</th>
      <th>Plovdiv</th>
      <td>84</td>
      <td>341,041</td>
      <td>31 December 2013</td>
      <td>341041</td>
    </tr>
    <tr>
      <th>Sofia</th>
      <td>14</td>
      <td>1,291,895</td>
      <td>14 December 2014</td>
      <td>1291895</td>
    </tr>
    <tr>
      <th>Varna</th>
      <td>85</td>
      <td>335,819</td>
      <td>31 December 2013</td>
      <td>335819</td>
    </tr>
    <tr>
      <th>Croatia</th>
      <th>Zagreb</th>
      <td>24</td>
      <td>790,017</td>
      <td>31 March 2011</td>
      <td>790017</td>
    </tr>
  </tbody>
</table>
</div>




```python
# partial indexing
df_pop3.loc["Sweden"]
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
      <th>Rank</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
      <th>NumericPopulation</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gothenburg</th>
      <td>53</td>
      <td>528,014</td>
      <td>31 March 2013</td>
      <td>528014</td>
    </tr>
    <tr>
      <th>Malmö</th>
      <td>102</td>
      <td>309,105</td>
      <td>31 March 2013</td>
      <td>309105</td>
    </tr>
    <tr>
      <th>Stockholm</th>
      <td>20</td>
      <td>909,976</td>
      <td>31 January 2014</td>
      <td>909976</td>
    </tr>
  </tbody>
</table>
</div>




```python
# complete indexing with tuple of all hierarchical indices
df_pop3.loc[("Sweden", "Gothenburg")]
```




    Rank                                  53
    Population                       528,014
    Date of census/estimate    31 March 2013
    NumericPopulation                 528014
    Name: (Sweden, Gothenburg), dtype: object




```python
# sort by column other than index
df_pop.set_index("City").sort_values(by=["State", "NumericPopulation"], 
                              ascending=[False, True]).head()
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
      <th>Rank</th>
      <th>State</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
      <th>NumericPopulation</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nottingham</th>
      <td>103</td>
      <td>United Kingdom</td>
      <td>308,735</td>
      <td>30 June 2012</td>
      <td>308735</td>
    </tr>
    <tr>
      <th>Wirral</th>
      <td>97</td>
      <td>United Kingdom</td>
      <td>320,229</td>
      <td>30 June 2012</td>
      <td>320229</td>
    </tr>
    <tr>
      <th>Coventry</th>
      <td>94</td>
      <td>United Kingdom</td>
      <td>323,132</td>
      <td>30 June 2012</td>
      <td>323132</td>
    </tr>
    <tr>
      <th>Wakefield</th>
      <td>91</td>
      <td>United Kingdom</td>
      <td>327,627</td>
      <td>30 June 2012</td>
      <td>327627</td>
    </tr>
    <tr>
      <th>Leicester</th>
      <td>87</td>
      <td>United Kingdom</td>
      <td>331,606</td>
      <td>30 June 2012</td>
      <td>331606</td>
    </tr>
  </tbody>
</table>
</div>




```python
# summarizing categorical data
city_counts = df_pop.State.value_counts()
city_counts.name = "# cities in top 105"
city_counts
```




    State
    Germany                     19
    United Kingdom              16
    Spain                       13
    Poland                      10
    Italy                       10
    France                       5
    Netherlands                  4
    Bulgaria                     3
    Sweden                       3
    Romania                      3
    Czech Republic               2
    Belgium                      2
    Greece                       2
    Denmark                      2
    Lithuania                    2
    Hungary                      1
    Austria                      1
    Croatia                      1
    Latvia                       1
    Finland                      1
    Portugal                     1
    Ireland                      1
    Estonia                      1
    Slovakia Slovak Republic     1
    Name: # cities in top 105, dtype: int64




```python
# how much population of all cities within a state?
df_pop3 = df_pop[["State", "City", "NumericPopulation"]].set_index(["State", "City"])
```


```python
df_pop4 = df_pop3.sum(level="State").sort_values(by="NumericPopulation", ascending=False)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[47], line 1
    ----> 1 df_pop4 = df_pop3.sum(level="State").sort_values(by="NumericPopulation", ascending=False)


    File ~/.local/lib/python3.11/site-packages/pandas/core/frame.py:11312, in DataFrame.sum(self, axis, skipna, numeric_only, min_count, **kwargs)
      11303 @doc(make_doc("sum", ndim=2))
      11304 def sum(
      11305     self,
       (...)
      11310     **kwargs,
      11311 ):
    > 11312     result = super().sum(axis, skipna, numeric_only, min_count, **kwargs)
      11313     return result.__finalize__(self, method="sum")


    File ~/.local/lib/python3.11/site-packages/pandas/core/generic.py:12070, in NDFrame.sum(self, axis, skipna, numeric_only, min_count, **kwargs)
      12062 def sum(
      12063     self,
      12064     axis: Axis | None = 0,
       (...)
      12068     **kwargs,
      12069 ):
    > 12070     return self._min_count_stat_function(
      12071         "sum", nanops.nansum, axis, skipna, numeric_only, min_count, **kwargs
      12072     )


    File ~/.local/lib/python3.11/site-packages/pandas/core/generic.py:12035, in NDFrame._min_count_stat_function(self, name, func, axis, skipna, numeric_only, min_count, **kwargs)
      12023 @final
      12024 def _min_count_stat_function(
      12025     self,
       (...)
      12032     **kwargs,
      12033 ):
      12034     assert name in ["sum", "prod"], name
    > 12035     nv.validate_func(name, (), kwargs)
      12037     validate_bool_kwarg(skipna, "skipna", none_allowed=False)
      12039     if axis is None:


    File ~/.local/lib/python3.11/site-packages/pandas/compat/numpy/function.py:416, in validate_func(fname, args, kwargs)
        413     return validate_stat_func(args, kwargs, fname=fname)
        415 validation_func = _validation_funcs[fname]
    --> 416 return validation_func(args, kwargs)


    File ~/.local/lib/python3.11/site-packages/pandas/compat/numpy/function.py:88, in CompatValidator.__call__(self, args, kwargs, fname, max_fname_arg_count, method)
         86     validate_kwargs(fname, kwargs, self.defaults)
         87 elif method == "both":
    ---> 88     validate_args_and_kwargs(
         89         fname, args, kwargs, max_fname_arg_count, self.defaults
         90     )
         91 else:
         92     raise ValueError(f"invalid validation method '{method}'")


    File ~/.local/lib/python3.11/site-packages/pandas/util/_validators.py:223, in validate_args_and_kwargs(fname, args, kwargs, max_fname_arg_count, compat_args)
        218         raise TypeError(
        219             f"{fname}() got multiple values for keyword argument '{key}'"
        220         )
        222 kwargs.update(args_dict)
    --> 223 validate_kwargs(fname, kwargs, compat_args)


    File ~/.local/lib/python3.11/site-packages/pandas/util/_validators.py:164, in validate_kwargs(fname, kwargs, compat_args)
        142 """
        143 Checks whether parameters passed to the **kwargs argument in a
        144 function `fname` are valid parameters as specified in `*compat_args`
       (...)
        161 map to the default values specified in `compat_args`
        162 """
        163 kwds = kwargs.copy()
    --> 164 _check_for_invalid_keys(fname, kwargs, compat_args)
        165 _check_for_default_values(fname, kwds, compat_args)


    File ~/.local/lib/python3.11/site-packages/pandas/util/_validators.py:138, in _check_for_invalid_keys(fname, kwargs, compat_args)
        136 if diff:
        137     bad_arg = next(iter(diff))
    --> 138     raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")


    TypeError: sum() got an unexpected keyword argument 'level'



```python
df_pop4.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[48], line 1
    ----> 1 df_pop4.head()


    NameError: name 'df_pop4' is not defined



```python
# alternative using groupby

df_pop5 = (df_pop.drop("Rank", axis=1)
                 .groupby("State").sum()
                 .sort_values(by="NumericPopulation", ascending=False))
df_pop5.head()
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
      <th>City</th>
      <th>Population</th>
      <th>Date of census/estimate</th>
      <th>NumericPopulation</th>
    </tr>
    <tr>
      <th>State</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United Kingdom</th>
      <td>London[2]BirminghamLeedsGlasgowSheffieldBradfo...</td>
      <td>8,615,2461,092,330757,655596,550557,382524,619...</td>
      <td>1 June 201430 June 201330 June 201231 December...</td>
      <td>16011877</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>BerlinHamburg[10]MunichCologneFrankfurtStuttga...</td>
      <td>3,437,9161,746,3421,407,8361,034,175701,350604...</td>
      <td>31 May 201430 December 201331 December 201331 ...</td>
      <td>15119548</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>MadridBarcelonaValenciaSevilleZaragozaMálagaMu...</td>
      <td>3,165,2351,602,386786,424696,676666,058566,913...</td>
      <td>1 January 20141 January 20141 January 20141 Ja...</td>
      <td>10041639</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>RomeMilanNaplesTurinPalermoGenoaBolognaFlorenc...</td>
      <td>2,872,0861,332,516989,845898,095677,015594,774...</td>
      <td>30 September 201430 September 201430 September...</td>
      <td>8764067</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>WarsawKrakówŁódźWrocławPoznańGdańskSzczecinByd...</td>
      <td>1,729,119760,700709,757632,432547,161460,35440...</td>
      <td>31 March 201431 March 201431 March 201431 Marc...</td>
      <td>6267409</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

city_counts.plot(kind='barh', ax=ax1)
ax1.set_xlabel("# cities in top 105")
df_pop5.NumericPopulation.plot(kind='barh', ax=ax2)
ax2.set_xlabel("Total pop. in top 105 cities")

fig.tight_layout()
#fig.savefig("ch12-state-city-counts-sum.pdf")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_53_0.png)
    


### Time series
* **Series** & **DataFrame** can support time series structures. Pandas has **DatetimeIndex** & **PeriodIndex** indexers to help.


```python
import datetime
```


```python
# generate date range. default frequency = daily.
pd.date_range("2015-1-1", periods=31)
```




    DatetimeIndex(['2015-01-01', '2015-01-02', '2015-01-03', '2015-01-04',
                   '2015-01-05', '2015-01-06', '2015-01-07', '2015-01-08',
                   '2015-01-09', '2015-01-10', '2015-01-11', '2015-01-12',
                   '2015-01-13', '2015-01-14', '2015-01-15', '2015-01-16',
                   '2015-01-17', '2015-01-18', '2015-01-19', '2015-01-20',
                   '2015-01-21', '2015-01-22', '2015-01-23', '2015-01-24',
                   '2015-01-25', '2015-01-26', '2015-01-27', '2015-01-28',
                   '2015-01-29', '2015-01-30', '2015-01-31'],
                  dtype='datetime64[ns]', freq='D')




```python
# generate datetime range (starting, ending, hourly)
pd.date_range("2015-1-1 00:00", "2015-1-1 12:00", freq="H")
```




    DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
                   '2015-01-01 02:00:00', '2015-01-01 03:00:00',
                   '2015-01-01 04:00:00', '2015-01-01 05:00:00',
                   '2015-01-01 06:00:00', '2015-01-01 07:00:00',
                   '2015-01-01 08:00:00', '2015-01-01 09:00:00',
                   '2015-01-01 10:00:00', '2015-01-01 11:00:00',
                   '2015-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq='H')




```python
# return instance of DatetimeIndex
ts1 = pd.Series(
    np.arange(31), 
    index=pd.date_range("2015-1-1", periods=31))

ts1.head()
```




    2015-01-01    0
    2015-01-02    1
    2015-01-03    2
    2015-01-04    3
    2015-01-05    4
    Freq: D, dtype: int64




```python
ts1["2015-1-3"]
```




    2




```python
ts1.index[2]
```




    Timestamp('2015-01-03 00:00:00')




```python
# time series elements
ts1.index[2].year, ts1.index[2].month, ts1.index[2].day
```




    (2015, 1, 3)




```python
# Timestamps = nanosecond resolution
# datetimes  = microsecond resolution
ts1.index[2].nanosecond
```




    0




```python
# conversion to datetime
ts1.index[2].to_pydatetime()
```




    datetime.datetime(2015, 1, 3, 0, 0)




```python
ts2 = pd.Series(np.random.rand(2), 
                index=[datetime.datetime(2015, 1, 1), 
                       datetime.datetime(2015, 2, 1)])
ts2
```




    2015-01-01    0.622530
    2015-02-01    0.918973
    dtype: float64




```python
# PeriodIndex - use to define sequences of time spans
periods = pd.PeriodIndex(
    [pd.Period('2015-01'), 
     pd.Period('2015-02'), 
     pd.Period('2015-03')])
```


```python
ts3 = pd.Series(np.random.rand(3), periods)
ts3
```




    2015-01    0.248811
    2015-02    0.850521
    2015-03    0.218363
    Freq: M, dtype: float64




```python
ts3.index
```




    PeriodIndex(['2015-01', '2015-02', '2015-03'], dtype='period[M]')




```python
ts2.to_period('M')
```




    2015-01    0.622530
    2015-02    0.918973
    Freq: M, dtype: float64




```python
pd.date_range("2015-1-1", periods=12, freq="M").to_period()
```




    PeriodIndex(['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06',
                 '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12'],
                dtype='period[M]')



### Temperature time series example
* indoor & outdoor datasets in TSV (tab-separated value) format
* each with 2 columns: UNIX timestamp, temperature (celsius)


```python
!head -n 5 temperature_outdoor_2014.tsv
```

    1388530986	4.380000
    1388531586	4.250000
    1388532187	4.190000
    1388532787	4.060000
    1388533388	4.060000



```python
df1 = pd.read_csv(
    'temperature_outdoor_2014.tsv', 
    delimiter="\t", 
    names=["time", "outdoor"])
```


```python
df2 = pd.read_csv(
    'temperature_indoor_2014.tsv', 
    delimiter="\t", 
    names=["time", "indoor"])
```


```python
df1.head()
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
      <th>time</th>
      <th>outdoor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1388530986</td>
      <td>4.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1388531586</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1388532187</td>
      <td>4.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1388532787</td>
      <td>4.06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1388533388</td>
      <td>4.06</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.head()
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
      <th>time</th>
      <th>indoor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1388530986</td>
      <td>21.94</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1388531586</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1388532187</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1388532787</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1388533388</td>
      <td>22.00</td>
    </tr>
  </tbody>
</table>
</div>



* Convert UNIX timestampes to date & time objects
* Also localize timestamps to a given timezone.


```python
df1.time = (pd.to_datetime(df1.time.values, unit="s")
              .tz_localize('UTC').tz_convert('Europe/Stockholm'))
```


```python
df1 = df1.set_index("time")
```


```python
df2.time = (pd.to_datetime(df2.time.values, unit="s")
              .tz_localize('UTC').tz_convert('Europe/Stockholm'))
```


```python
df2 = df2.set_index("time")
```


```python
df1.head()
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
      <th>outdoor</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01-01 00:03:06+01:00</th>
      <td>4.38</td>
    </tr>
    <tr>
      <th>2014-01-01 00:13:06+01:00</th>
      <td>4.25</td>
    </tr>
    <tr>
      <th>2014-01-01 00:23:07+01:00</th>
      <td>4.19</td>
    </tr>
    <tr>
      <th>2014-01-01 00:33:07+01:00</th>
      <td>4.06</td>
    </tr>
    <tr>
      <th>2014-01-01 00:43:08+01:00</th>
      <td>4.06</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.index[0]
```




    Timestamp('2014-01-01 00:03:06+0100', tz='Europe/Stockholm')




```python
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df1.plot(ax=ax)
df2.plot(ax=ax)

fig.tight_layout()
#fig.savefig("ch12-timeseries-temperature-2014.pdf")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_83_0.png)
    



```python
# almost 50K datapoints across 2014
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 49548 entries, 2014-01-01 00:03:06+01:00 to 2014-12-30 23:56:35+01:00
    Data columns (total 1 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   outdoor  49548 non-null  float64
    dtypes: float64(1)
    memory usage: 774.2 KB


* Common use case: selecting/extracting portions of a dataset
* In this case, January data


```python
df1_jan = df1[
    (df1.index > "2014-1-1") & 
    (df1.index < "2014-2-1")]
```


```python
df1.index < "2014-2-1"
```




    array([ True,  True,  True, ..., False, False, False])




```python
df1_jan.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 4452 entries, 2014-01-01 00:03:06+01:00 to 2014-01-31 23:56:58+01:00
    Data columns (total 1 columns):
    outdoor    4452 non-null float64
    dtypes: float64(1)
    memory usage: 69.6 KB



```python
df2_jan = df2["2014-1-1":"2014-1-31"]
```


```python
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

df1_jan.plot(ax=ax)
df2_jan.plot(ax=ax)

fig.tight_layout()
#fig.savefig("ch12-timeseries-selected-month.pdf")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_90_0.png)
    


* Grouping data (in this case, by month)


```python
df1_month = df1.reset_index()
```


```python
df1_month["month"] = df1_month.time.apply(lambda x: x.month)
```


```python
df1_month.head()
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
      <th>time</th>
      <th>outdoor</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-01-01 00:03:06+01:00</td>
      <td>4.38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-01-01 00:13:06+01:00</td>
      <td>4.25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-01-01 00:23:07+01:00</td>
      <td>4.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-01-01 00:33:07+01:00</td>
      <td>4.06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-01-01 00:43:08+01:00</td>
      <td>4.06</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



* Now group DataFrame by new month field & aggregate grouped values with mean()


```python
df1_month = df1_month.groupby("month").aggregate(np.mean)
```


```python
df2_month = df2.reset_index()
```


```python
df2_month["month"] = df2_month.time.apply(lambda x: x.month)
```


```python
df2_month = df2_month.groupby("month").aggregate(np.mean)
```


```python
df_month = df1_month.join(df2_month)
```


```python
df_month.head(3)
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
      <th>outdoor</th>
      <th>indoor</th>
    </tr>
    <tr>
      <th>month</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-1.776646</td>
      <td>19.862590</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.231613</td>
      <td>20.231507</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.615437</td>
      <td>19.597748</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_month = pd.concat([df.to_period("M").groupby(level=0).mean() for df in [df1, df2]], axis=1)
```


```python
df_month.head(3)
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
      <th>outdoor</th>
      <th>indoor</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01</th>
      <td>-1.776646</td>
      <td>19.862590</td>
    </tr>
    <tr>
      <th>2014-02</th>
      <td>2.231613</td>
      <td>20.231507</td>
    </tr>
    <tr>
      <th>2014-03</th>
      <td>4.615437</td>
      <td>19.597748</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df_month.plot(kind='bar', ax=axes[0])
df_month.plot(kind='box', ax=axes[1])

fig.tight_layout()
#fig.savefig("ch12-grouped-by-month.pdf")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_104_0.png)
    


### Resampling


```python
df_month
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
      <th>outdoor</th>
      <th>indoor</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01</th>
      <td>-1.776646</td>
      <td>19.862590</td>
    </tr>
    <tr>
      <th>2014-02</th>
      <td>2.231613</td>
      <td>20.231507</td>
    </tr>
    <tr>
      <th>2014-03</th>
      <td>4.615437</td>
      <td>19.597748</td>
    </tr>
    <tr>
      <th>2014-04</th>
      <td>8.105193</td>
      <td>22.149754</td>
    </tr>
    <tr>
      <th>2014-05</th>
      <td>12.261396</td>
      <td>26.332160</td>
    </tr>
    <tr>
      <th>2014-06</th>
      <td>15.586955</td>
      <td>28.687491</td>
    </tr>
    <tr>
      <th>2014-07</th>
      <td>20.780314</td>
      <td>30.605333</td>
    </tr>
    <tr>
      <th>2014-08</th>
      <td>16.494823</td>
      <td>28.099068</td>
    </tr>
    <tr>
      <th>2014-09</th>
      <td>12.823905</td>
      <td>26.950366</td>
    </tr>
    <tr>
      <th>2014-10</th>
      <td>9.352000</td>
      <td>23.379460</td>
    </tr>
    <tr>
      <th>2014-11</th>
      <td>4.992142</td>
      <td>20.610365</td>
    </tr>
    <tr>
      <th>2014-12</th>
      <td>-0.058940</td>
      <td>16.465674</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 49548 entries, 2014-01-01 00:03:06+01:00 to 2014-12-30 23:56:35+01:00
    Data columns (total 1 columns):
    outdoor    49548 non-null float64
    dtypes: float64(1)
    memory usage: 774.2 KB



```python
# resample(): 1st argument = string that represents a new data period.
# ex: "H" = hourly

df1_hour = df1.resample("H").mean()
df1_hour.columns = ["outdoor (hourly avg.)"]

df1_day = df1.resample("D").mean()
df1_day.columns = ["outdoor (daily avg.)"]

df1_week = df1.resample("W").mean()
df1_week.columns = ["outdoor (weekly avg.)"]

df1_month = df1.resample("M").mean()
df1_month.columns = ["outdoor (monthly avg.)"]

df_diff = (df1.resample("D").mean().outdoor - df2.resample("D").mean().indoor)
```


```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

df1_hour.plot(ax=ax1, alpha=0.25)
df1_day.plot(ax=ax1)
df1_week.plot(ax=ax1)
df1_month.plot(ax=ax1)

df_diff.plot(ax=ax2)
ax2.set_title("temperature difference between outdoor and indoor")

fig.tight_layout()
#fig.savefig("ch12-timeseries-resampled.pdf")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_109_0.png)
    


* Resampling to 5-minute frequency with three aggregation methods
* mean, ffill (forward fill), bfill (backward fill)
* Some values are not filled (NaN) based on aggregation method.


```python
pd.concat(
    [df1.resample("5min").mean().rename(columns={"outdoor": 'None'}),
     df1.resample("5min").ffill().rename(columns={"outdoor": 'ffill'}),
     df1.resample("5min").bfill().rename(columns={"outdoor": 'bfill'})],
    axis=1).head()
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
      <th>None</th>
      <th>ffill</th>
      <th>bfill</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01-01 00:00:00+01:00</th>
      <td>4.38</td>
      <td>NaN</td>
      <td>4.38</td>
    </tr>
    <tr>
      <th>2014-01-01 00:05:00+01:00</th>
      <td>NaN</td>
      <td>4.38</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>2014-01-01 00:10:00+01:00</th>
      <td>4.25</td>
      <td>4.38</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>2014-01-01 00:15:00+01:00</th>
      <td>NaN</td>
      <td>4.25</td>
      <td>4.19</td>
    </tr>
    <tr>
      <th>2014-01-01 00:20:00+01:00</th>
      <td>4.19</td>
      <td>4.25</td>
      <td>4.19</td>
    </tr>
  </tbody>
</table>
</div>



# Seaborn statistical visualization library
- Built on top of Matplotlib
- Distribution plots, kernel density plots, joint distributions, factor plots, heatmaps, facet plots, many more.
- Much better color usage (aesthetics)


```python
import seaborn as sns
```


```python
sns.set(style="darkgrid")
```


```python
#sns.set(style="whitegrid")
```


```python
df1 = pd.read_csv(
    'temperature_outdoor_2014.tsv', 
    delimiter="\t", 
    names=["time", "outdoor"])

df1.time = pd.to_datetime(
    df1.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')

df1 = df1.set_index("time").resample("10min").mean()

df2 = pd.read_csv(
    'temperature_indoor_2014.tsv', 
    delimiter="\t", 
    names=["time", "indoor"])

df2.time = pd.to_datetime(
    df2.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')

df2 = df2.set_index("time").resample("10min").mean()

df_temp = pd.concat([df1, df2], axis=1)
```


```python
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df_temp.resample("D").plot(y=["outdoor", "indoor"], ax=ax)
fig.tight_layout()
#fig.savefig("ch12-seaborn-plot.pdf")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_117_0.png)
    



```python
# distribuion (histogram) plots, outdoor & indoor temp data
sns.distplot(df_temp.to_period("M")["outdoor"]["2014-04"].dropna().values, bins=50);
sns.distplot(df_temp.to_period("M")["indoor"]["2014-04"].dropna().values, bins=50);

plt.savefig("ch12-seaborn-distplot.pdf")
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_118_1.png)
    



```python
df_temp.resample("H")["outdoor"].describe
```




    <bound method SeriesGroupBy.describe of <pandas.core.groupby.groupby.SeriesGroupBy object at 0x7f354ee2eac8>>




```python
# joint distribution plot - indoor/outdoor temp correlation, resampled to hourly averages
# TODO: resolve "non-tuple sequence for multi-d indexing is deprecated" warning

with sns.axes_style("white"):
    sns.jointplot(df_temp.resample("H").mean()["outdoor"].values,
                  df_temp.resample("H").mean()["indoor"].values, kind="hex");
    
#plt.savefig("ch12-seaborn-jointplot.pdf")
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_120_1.png)
    



```python
# Kernel Density Estimation (KDE) plot
# TODO: resolve "non-tuple sequence for multi-d indexing is deprecated" warning

sns.kdeplot(df_temp.resample("H").mean()["outdoor"].dropna().values,
            df_temp.resample("H").mean()["indoor"].dropna().values, shade=False);

#plt.savefig("ch12-seaborn-kdeplot.pdf")
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_121_1.png)
    



```python
# category data helpers:
# box plot (viz for descriptive stats: min, max, median, quartiles)
# violin plot (box plot with KDE data shown by box plot width)
# TODO: resolve "non-tuple sequence for multi-d indexing is deprecated" warning

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

sns.boxplot(df_temp.dropna(), ax=ax1, palette="pastel")
sns.violinplot(df_temp.dropna(), ax=ax2, palette="pastel")

fig.tight_layout()
#fig.savefig("ch12-seaborn-boxplot-violinplot.pdf")
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_122_1.png)
    



```python
#violin plot: outdoor temp data partitioned by month
# shows distribution of temp for each month

sns.violinplot(
    x=df_temp.dropna().index.month, 
    y=df_temp.dropna().outdoor, 
    color="skyblue");

#plt.savefig("ch12-seaborn-violinplot.pdf")
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_123_1.png)
    



```python
# heatmaps: useful for categorical data with large # of categories.

df_temp["month"] = df_temp.index.month
df_temp["hour"] = df_temp.index.hour
```


```python
df_temp.head()
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
      <th>outdoor</th>
      <th>indoor</th>
      <th>month</th>
      <th>hour</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01-01 00:00:00+01:00</th>
      <td>4.38</td>
      <td>21.94</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2014-01-01 00:10:00+01:00</th>
      <td>4.25</td>
      <td>22.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2014-01-01 00:20:00+01:00</th>
      <td>4.19</td>
      <td>22.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2014-01-01 00:30:00+01:00</th>
      <td>4.06</td>
      <td>22.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2014-01-01 00:40:00+01:00</th>
      <td>4.06</td>
      <td>22.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
table = pd.pivot_table(
    df_temp, 
    values='outdoor', 
    index=['month'], 
    columns=['hour'], 
    aggfunc=np.mean); table
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
      <th>hour</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
    <tr>
      <th>month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-1.692312</td>
      <td>-1.750162</td>
      <td>-1.826649</td>
      <td>-1.879086</td>
      <td>-1.922527</td>
      <td>-1.968065</td>
      <td>-2.020914</td>
      <td>-2.035806</td>
      <td>-2.101774</td>
      <td>-2.001022</td>
      <td>...</td>
      <td>-1.457849</td>
      <td>-1.696935</td>
      <td>-1.814194</td>
      <td>-1.812258</td>
      <td>-1.853297</td>
      <td>-1.898432</td>
      <td>-1.839730</td>
      <td>-1.806486</td>
      <td>-1.854462</td>
      <td>-1.890811</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.613690</td>
      <td>1.521190</td>
      <td>1.479405</td>
      <td>1.464371</td>
      <td>1.506407</td>
      <td>1.485595</td>
      <td>1.499167</td>
      <td>1.516946</td>
      <td>1.669226</td>
      <td>2.067725</td>
      <td>...</td>
      <td>3.573593</td>
      <td>3.360741</td>
      <td>2.939390</td>
      <td>2.501607</td>
      <td>2.357425</td>
      <td>2.236190</td>
      <td>2.204458</td>
      <td>2.137619</td>
      <td>2.024671</td>
      <td>1.896190</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.192366</td>
      <td>2.866774</td>
      <td>2.628000</td>
      <td>2.524140</td>
      <td>2.384140</td>
      <td>2.235538</td>
      <td>2.243387</td>
      <td>2.622258</td>
      <td>3.419301</td>
      <td>4.466290</td>
      <td>...</td>
      <td>7.790323</td>
      <td>7.930914</td>
      <td>7.595892</td>
      <td>6.770914</td>
      <td>5.731508</td>
      <td>4.983784</td>
      <td>4.437419</td>
      <td>4.022312</td>
      <td>3.657903</td>
      <td>3.407258</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.832738</td>
      <td>5.336012</td>
      <td>4.926667</td>
      <td>4.597059</td>
      <td>4.380000</td>
      <td>4.109769</td>
      <td>4.123699</td>
      <td>4.741437</td>
      <td>5.878035</td>
      <td>7.272299</td>
      <td>...</td>
      <td>12.175556</td>
      <td>12.500059</td>
      <td>12.494483</td>
      <td>12.361156</td>
      <td>11.989240</td>
      <td>10.454881</td>
      <td>8.857619</td>
      <td>7.712619</td>
      <td>6.974762</td>
      <td>6.293512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.792204</td>
      <td>9.369351</td>
      <td>9.009839</td>
      <td>8.670914</td>
      <td>8.463387</td>
      <td>8.446919</td>
      <td>8.772324</td>
      <td>9.562742</td>
      <td>10.723622</td>
      <td>12.047717</td>
      <td>...</td>
      <td>15.542581</td>
      <td>15.744624</td>
      <td>15.784839</td>
      <td>15.799570</td>
      <td>17.009892</td>
      <td>15.685161</td>
      <td>13.632796</td>
      <td>12.216290</td>
      <td>11.291237</td>
      <td>10.622849</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13.209556</td>
      <td>12.792889</td>
      <td>12.382889</td>
      <td>11.967889</td>
      <td>11.735778</td>
      <td>11.886667</td>
      <td>12.503778</td>
      <td>13.338167</td>
      <td>14.343444</td>
      <td>15.665475</td>
      <td>...</td>
      <td>18.630556</td>
      <td>18.866292</td>
      <td>18.680611</td>
      <td>18.529832</td>
      <td>20.057877</td>
      <td>18.853389</td>
      <td>16.969777</td>
      <td>15.675111</td>
      <td>14.658778</td>
      <td>13.898167</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17.956344</td>
      <td>17.348641</td>
      <td>16.793152</td>
      <td>16.309892</td>
      <td>16.001559</td>
      <td>15.986774</td>
      <td>16.506613</td>
      <td>17.478226</td>
      <td>18.850054</td>
      <td>20.533763</td>
      <td>...</td>
      <td>24.598441</td>
      <td>25.030000</td>
      <td>24.869194</td>
      <td>24.764409</td>
      <td>26.155161</td>
      <td>24.896505</td>
      <td>22.550269</td>
      <td>20.882649</td>
      <td>19.699022</td>
      <td>18.822634</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14.498205</td>
      <td>13.960128</td>
      <td>13.555128</td>
      <td>12.995641</td>
      <td>12.651410</td>
      <td>12.485974</td>
      <td>12.680130</td>
      <td>13.403506</td>
      <td>14.578780</td>
      <td>16.170833</td>
      <td>...</td>
      <td>20.473810</td>
      <td>20.292381</td>
      <td>20.328795</td>
      <td>19.642436</td>
      <td>19.373846</td>
      <td>18.713462</td>
      <td>17.034872</td>
      <td>15.843590</td>
      <td>15.146154</td>
      <td>14.596667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11.133000</td>
      <td>10.725667</td>
      <td>10.362444</td>
      <td>9.976833</td>
      <td>9.729333</td>
      <td>9.503944</td>
      <td>9.357500</td>
      <td>9.689778</td>
      <td>10.600778</td>
      <td>11.829106</td>
      <td>...</td>
      <td>16.336983</td>
      <td>16.828268</td>
      <td>17.031056</td>
      <td>16.786983</td>
      <td>15.853556</td>
      <td>14.534637</td>
      <td>13.350444</td>
      <td>12.545278</td>
      <td>11.954190</td>
      <td>11.399056</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8.602011</td>
      <td>8.490598</td>
      <td>8.382486</td>
      <td>8.257097</td>
      <td>8.166774</td>
      <td>8.140054</td>
      <td>8.140161</td>
      <td>8.148333</td>
      <td>8.410914</td>
      <td>9.054946</td>
      <td>...</td>
      <td>11.330323</td>
      <td>11.189194</td>
      <td>10.836865</td>
      <td>10.361568</td>
      <td>9.781022</td>
      <td>9.373441</td>
      <td>9.134570</td>
      <td>8.956505</td>
      <td>8.820270</td>
      <td>8.623297</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4.847111</td>
      <td>4.765922</td>
      <td>4.815642</td>
      <td>4.773240</td>
      <td>4.809611</td>
      <td>4.785833</td>
      <td>4.741222</td>
      <td>4.739778</td>
      <td>4.794500</td>
      <td>4.965389</td>
      <td>...</td>
      <td>5.526034</td>
      <td>5.342753</td>
      <td>5.081250</td>
      <td>5.056629</td>
      <td>4.959106</td>
      <td>4.868111</td>
      <td>4.833333</td>
      <td>4.774389</td>
      <td>4.720722</td>
      <td>4.699722</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.366369</td>
      <td>-0.390556</td>
      <td>-0.447374</td>
      <td>-0.370111</td>
      <td>-0.353128</td>
      <td>-0.319832</td>
      <td>-0.358667</td>
      <td>-0.410278</td>
      <td>-0.483167</td>
      <td>-0.344667</td>
      <td>...</td>
      <td>0.738944</td>
      <td>0.367056</td>
      <td>0.152167</td>
      <td>-0.106111</td>
      <td>-0.182500</td>
      <td>-0.244167</td>
      <td>-0.290000</td>
      <td>-0.305333</td>
      <td>-0.302778</td>
      <td>-0.325642</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 24 columns</p>
</div>




```python
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.heatmap(table, ax=ax);

fig.tight_layout()
#fig.savefig("ch12-seaborn-heatmap.pdf")
```


    
![png](ch12-data-analysis-pandas-seaborn_files/ch12-data-analysis-pandas-seaborn_127_0.png)
    

