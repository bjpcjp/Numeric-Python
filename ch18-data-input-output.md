# Data Input/Output Options

- __Comma-Separated Values (CSV)__
- __Hierarchical Data Format (HDF5)__
- __h5py__ (files, groups, datasets, attributes)
- __PyTables__
- __Pandas HDFStore__
- __JSON__
- __Serialization__

* data classes:
    * structured vs unstructured
    * categorical (finite set) vs ordinal (ordered) vs numerical (continuous/discrete)
* should consider: [Blaze](http://blaze.pydata.org/en/latest) for high-level, multi-format API for data I/O

## Imports


```python
from __future__ import print_function
```


```python
import numpy as np
np.random.seed(0)
```


```python
import pandas as pd
```


```python
import csv
import json
import h5py
import tables
import pickle

# python3: import _pickle as cPickle
import _pickle as cPickle

# conda install msgpack-python
import msgpack
```

    /home/bjpcjp/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


### CSV

- first: create some example CSV data (hockey player stats) & save it to disk


```python
%%writefile ch18-playerstats-2013-2014.csv
# 2013-2014 / Regular Season / All Skaters / Summary / Points
Rank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%
1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5
2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0
3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9
4,Tyler Seguin,DAL,C,80,37,47,84,+16,18,11,25,0,0,8,0,294,12.6,19:20,23.4,41.5
5,Corey Perry,ANA,R,81,43,39,82,+32,65,8,18,0,0,9,1,280,15.4,19:28,23.2,36.0
```

    Overwriting ch18-playerstats-2013-2014.csv



```python
%%writefile ch18-playerstats-2013-2014-top30.csv
# 2013-2014 / Regular Season / All Skaters / Summary / Points
Rank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%
1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5
2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0
3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9
4,Tyler Seguin,DAL,C,80,37,47,84,+16,18,11,25,0,0,8,0,294,12.6,19:20,23.4,41.5
5,Corey Perry,ANA,R,81,43,39,82,+32,65,8,18,0,0,9,1,280,15.4,19:28,23.2,36.0
6,Phil Kessel,TOR,R,82,37,43,80,-5,27,8,20,0,0,6,0,305,12.1,20:39,24.5,14.3
7,Taylor Hall,EDM,L,75,27,53,80,-15,44,7,17,0,1,1,1,250,10.8,20:00,25.4,45.7
8,Alex Ovechkin,WSH,L,78,51,28,79,-35,48,24,39,0,1,10,3,386,13.2,20:32,21.8,66.7
9,Joe Pavelski,SJS,C,82,41,38,79,+23,32,16,31,1,2,3,0,225,18.2,19:51,27.1,56.0
10,Jamie Benn,DAL,L,81,34,45,79,+21,64,5,19,1,3,3,1,279,12.2,19:09,25.0,52.8
11,Nicklas Backstrom,WSH,C,82,18,61,79,-20,54,6,44,1,1,1,0,196,9.2,19:48,23.3,50.4
12,Patrick Sharp,CHI,L,82,34,44,78,+13,40,10,25,0,0,3,1,313,10.9,18:53,22.7,54.6
13,Joe Thornton,SJS,C,82,11,65,76,+20,32,2,19,0,1,3,1,122,9.0,18:55,26.3,56.1
14,Erik Karlsson,OTT,D,82,20,54,74,-15,36,5,31,0,0,1,0,257,7.8,27:04,28.6,0.0
15,Evgeni Malkin,PIT,C,60,23,49,72,+10,62,7,30,0,0,3,0,191,12.0,20:03,21.4,48.8
16,Patrick Marleau,SJS,L,82,33,37,70,+0,18,11,23,2,2,4,0,285,11.6,20:31,27.3,52.9
17,Anze Kopitar,LAK,C,82,29,41,70,+34,24,10,23,0,0,9,2,200,14.5,20:53,25.4,53.3
18,Matt Duchene,COL,C,71,23,47,70,+8,19,5,17,0,0,6,1,217,10.6,18:29,22.0,50.3
19,Martin St. Louis,"TBL, NYR",R,81,30,39,69,+13,10,9,21,1,2,5,1,204,14.7,20:56,25.7,40.7
20,Patrick Kane,CHI,R,69,29,40,69,+7,22,10,25,0,0,6,0,227,12.8,19:36,22.9,50.0
21,Blake Wheeler,WPG,R,82,28,41,69,+4,63,8,19,0,0,4,2,225,12.4,18:41,24.0,37.5
22,Kyle Okposo,NYI,R,71,27,42,69,-9,51,5,15,0,0,4,1,195,13.8,20:26,22.2,47.5
23,David Krejci,BOS,C,80,19,50,69,+39,28,3,19,0,0,6,1,169,11.2,19:07,21.3,51.2
24,Chris Kunitz,PIT,L,78,35,33,68,+25,66,13,22,0,0,8,0,218,16.1,19:09,22.2,75.0
25,Jonathan Toews,CHI,C,76,28,40,68,+26,34,5,15,3,5,5,0,193,14.5,20:28,25.9,57.2
26,Thomas Vanek,"BUF, NYI, MTL",L,78,27,41,68,+7,46,8,18,0,0,4,0,248,10.9,19:21,21.6,43.5
27,Jaromir Jagr,NJD,R,82,24,43,67,+16,46,5,17,0,0,6,1,231,10.4,19:09,22.8,0.0
28,John Tavares,NYI,C,59,24,42,66,-6,40,8,25,0,0,4,0,188,12.8,21:14,22.3,49.1
29,Jason Spezza,OTT,C,75,23,43,66,-26,46,9,22,0,0,5,0,223,10.3,18:12,23.8,54.0
30,Jordan Eberle,EDM,R,80,28,37,65,-11,18,7,20,1,1,4,1,200,14.0,19:32,25.4,38.1
```

    Overwriting ch18-playerstats-2013-2014-top30.csv



```python
# let's see if file contents are as expected
!head -n 5 ch18-playerstats-2013-2014-top30.csv
```

    # 2013-2014 / Regular Season / All Skaters / Summary / Points
    Rank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%
    1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5
    2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0
    3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9


* Parsed row values will be read as strings, even if values represent numbers.
* Numpy __loadtxt__ and __savetxt__ are good for handling numerical arrays on disk.


```python
data = np.random.randn(100,3)
np.savetxt("data.csv", data, delimiter=",", header="x,y,z", comments="random x,y,z coords\n")
```


```python
!head -n 5 data.csv
```

    random x,y,z coords
    x,y,z
    1.764052345967664026e+00,4.001572083672232938e-01,9.787379841057392005e-01
    2.240893199201457797e+00,1.867557990149967484e+00,-9.772778798764110153e-01
    9.500884175255893682e-01,-1.513572082976978872e-01,-1.032188517935578448e-01



```python
# Read data back into NumPy array
data_load = np.loadtxt("data.csv", skiprows=2, delimiter=",")
# and check for equality
(data == data_load).all()
```




    True




```python
# by default, loadtxt converts all fields into float64 values
data_load[1,:]
```




    array([ 2.2408932 ,  1.86755799, -0.97727788])




```python
data_load.dtype
```




    dtype('float64')



* Need to explicitly set a dtype if reading non-numerical CSV data.
* Otherwise NumPy will barf.
* dtype=bytes, or str, or object, will return unparsed values.


```python
data = np.loadtxt(
    "ch18-playerstats-2013-2014.csv", 
    skiprows=2, delimiter=",", dtype=bytes)

data[0][1:6]
```




    array([b'Sidney Crosby', b'PIT', b'C', b'80', b'36'], dtype='|S13')




```python
# read selected columns:
np.loadtxt("ch18-playerstats-2013-2014.csv", 
           skiprows=2, delimiter=",", usecols=[6,7,8])
```




    array([[ 68., 104.,  18.],
           [ 56.,  87.,  28.],
           [ 58.,  86.,   7.],
           [ 47.,  84.,  16.],
           [ 39.,  82.,  32.]])




```python
# A 3rd method: Pandas read_csv()
df = pd.read_csv("ch18-playerstats-2013-2014.csv", 
                 skiprows=1)
```


```python
df = df.set_index("Rank")
```


```python
df[["Player", "GP", "G", "A", "P"]]
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
      <th>Player</th>
      <th>GP</th>
      <th>G</th>
      <th>A</th>
      <th>P</th>
    </tr>
    <tr>
      <th>Rank</th>
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
      <td>Sidney Crosby</td>
      <td>80</td>
      <td>36</td>
      <td>68</td>
      <td>104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ryan Getzlaf</td>
      <td>77</td>
      <td>31</td>
      <td>56</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Claude Giroux</td>
      <td>82</td>
      <td>28</td>
      <td>58</td>
      <td>86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tyler Seguin</td>
      <td>80</td>
      <td>37</td>
      <td>47</td>
      <td>84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Corey Perry</td>
      <td>81</td>
      <td>43</td>
      <td>39</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use info() to see the dtype of each parsed column
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5 entries, 1 to 5
    Data columns (total 20 columns):
    Player      5 non-null object
    Team        5 non-null object
    Pos         5 non-null object
    GP          5 non-null int64
    G           5 non-null int64
    A           5 non-null int64
    P           5 non-null int64
    +/-         5 non-null int64
    PIM         5 non-null int64
    PPG         5 non-null int64
    PPP         5 non-null int64
    SHG         5 non-null int64
    SHP         5 non-null int64
    GW          5 non-null int64
    OT          5 non-null int64
    S           5 non-null int64
    S%          5 non-null float64
    TOI/GP      5 non-null object
    Shift/GP    5 non-null float64
    FO%         5 non-null float64
    dtypes: float64(3), int64(13), object(4)
    memory usage: 840.0+ bytes



```python
# writing to CSV files using dataframes:
df[["Player", "GP", "G", "A", "P"]].to_csv("ch18-playerstats-2013-2014-subset.csv")
```


```python
!head -n 5 ch18-playerstats-2013-2014-subset.csv
```

    Rank,Player,GP,G,A,P
    1,Sidney Crosby,80,36,68,104
    2,Ryan Getzlaf,77,31,56,87
    3,Claude Giroux,82,28,58,86
    4,Tyler Seguin,80,37,47,84


### h5py
* used for numerical data store
* hierarchical format - orgs datasets within files: "groups" and "datasets"
* groups & datasets can contain "attributes" (metadata)
* Python libraries: h5py & PyTables


```python
import h5py
```

* file modes: "w" (create new file; truncate if exists), "r" (read-only; file must exist), "w-" (create new file; error if exists), "r+" (read-write; file must exist), "a" (read-write; create if needed)


```python
# create new read-write file
f = h5py.File("ch18-data.h5", "w")
f.mode
```




    'r+'




```python
f.flush()
f.close()
```

### Groups

* File object creates both file handle and a "root group" object.
* group name accessible via 'name'. root is '/'


```python
f = h5py.File("ch18-data.h5", "w")
f.name
```




    '/'




```python
# create hierarchical subgroups.
grp1      = f.create_group("experiment1")
grp2_meas = f.create_group("experiment2/measurement")
grp2_sim  = f.create_group("experiment2/simulation")

grp1.name, grp2_meas.name, grp2_sim.name
```




    ('/experiment1', '/experiment2/measurement', '/experiment2/simulation')




```python
# group access
f["/experiment1"]
```




    <HDF5 group "/experiment1" (0 members)>




```python
f["/experiment2/simulation"]
```




    <HDF5 group "/experiment2/simulation" (0 members)>




```python
grp_expr2 = f["/experiment2"]
```


```python
grp_expr2['simulation']
```




    <HDF5 group "/experiment2/simulation" (0 members)>




```python
# keys = names of subgroups & datasets within a group
list(f.keys())
```




    ['experiment1', 'experiment2']




```python
# items = tuples of (name, value) for each entity in each group
list(f.items())
```




    [('experiment1', <HDF5 group "/experiment1" (0 members)>),
     ('experiment2', <HDF5 group "/experiment2" (2 members)>)]




```python
# traverse group hierarchy
f.visit(lambda x: print(x))
```

    experiment1
    experiment2
    experiment2/measurement
    experiment2/simulation



```python
# traverse group hierarchy with item & item name accessible in arg
f.visititems(
    lambda name, 
    value: print(name, value))
```

    experiment1 <HDF5 group "/experiment1" (0 members)>
    experiment2 <HDF5 group "/experiment2" (2 members)>
    experiment2/measurement <HDF5 group "/experiment2/measurement" (0 members)>
    experiment2/simulation <HDF5 group "/experiment2/simulation" (0 members)>



```python
# membership testing
"experiment1" in f
```




    True




```python
"simulation" in f["experiment2"]
```




    True




```python
"experiment3" in f
```




    False




```python
f.flush()
```


```python
# h5ls = command-line tool for viewing HDF5 contents
# !h5ls ch18-data.h5
```

### HDF5 datasets
* two main methods to create a dataset in an HDF5 file:
    - easiest: assign NumPy array to an item in an HDF5 group (dictionary index syntax)
    - use __create_dataset__ method.


```python
data1 = np.arange(10)
data2 = np.random.randn(100, 100)
```


```python
f["array1"]                         = data1
f["/experiment2/measurement/meas1"] = data2
```


```python
# verify data was save correctly using visititems

f.visititems(
    lambda name, value: print(name, value))
```

    array1 <HDF5 dataset "array1": shape (10,), type "<i8">
    experiment1 <HDF5 group "/experiment1" (0 members)>
    experiment2 <HDF5 group "/experiment2" (2 members)>
    experiment2/measurement <HDF5 group "/experiment2/measurement" (1 members)>
    experiment2/measurement/meas1 <HDF5 dataset "meas1": shape (100, 100), type "<f8">
    experiment2/simulation <HDF5 group "/experiment2/simulation" (0 members)>



```python
# to retrieve array1 dataset (in root group)
# array1 is a Dataset object, not a NumPy array
ds = f["array1"]
ds 
```




    <HDF5 dataset "array1": shape (10,), type "<i8">




```python
ds.name, ds.dtype, ds.shape, ds.len()
```




    ('/array1', dtype('int64'), (10,), 10)




```python
ds.value
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# go deeper into hierarchy
ds = f["/experiment2/measurement/meas1"]
ds
```




    <HDF5 dataset "meas1": shape (100, 100), type "<f8">




```python
ds.dtype, ds.shape, ds.len
```




    (dtype('float64'),
     (100, 100),
     <bound method Dataset.len of <HDF5 dataset "meas1": shape (100, 100), type "<f8">>)




```python
# alternative syntax using [...]

data_full = ds[...]; data_full
```




    array([[-1.30652685,  1.65813068, -0.11816405, ...,  1.14110187,
             1.46657872,  0.85255194],
           [-0.59865394, -1.11589699,  0.76666318, ..., -0.51423397,
            -1.01804188, -0.07785476],
           [ 0.38273243, -0.03424228,  1.09634685, ..., -0.21673147,
            -0.9301565 , -0.17858909],
           ...,
           [-0.20211703, -0.833231  ,  1.73360025, ...,  0.77025427,
            -0.08612658, -0.85766795],
           [ 0.6391736 , -0.24720034,  0.23337957, ...,  0.17974832,
             0.26792302,  0.7701867 ],
           [ 1.31951239, -0.42585313,  0.09323029, ..., -0.51270866,
            -0.44602375,  1.89001412]])




```python
type(data_full), data_full.shape
```




    (numpy.ndarray, (100, 100))




```python
# retrieve only first column (a 100 element array)
data_col = ds[:, 0]
data_col.shape
```




    (100,)




```python
# Dataset objects support strided indexing:
ds[10:20:3, 10:20:3]
```




    array([[ 0.60270766, -0.34804638, -0.813596  , -1.29737966],
           [ 0.91320192, -1.06343294,  0.22734595,  0.52759738],
           [ 1.25774422, -0.32775492,  1.4849256 ,  0.28005786],
           [-0.84907287, -0.30000358,  1.79691852, -0.19871506]])




```python
# Dataset objects support "fancy" indexing:
ds[[1,2,3], :].shape
```




    (3, 100)




```python
# Boolean masking support
mask = ds[:, 0] > 2.0
```


```python
mask.shape, mask.dtype
```




    ((100,), dtype('bool'))




```python
# Single out first 5 columns (index :5 on 2nd axis) for each row
# whose 1st column value is larger than 2.
ds[mask, :5]
```




    array([[ 2.04253623, -0.91946118,  0.11467003, -0.1374237 ,  1.36552692],
           [ 2.1041854 ,  0.22725706, -1.1291663 , -0.28133197, -0.7394167 ],
           [ 2.05689385,  0.18041971, -0.06670925, -0.02835398,  0.48480475]])



### Creating empty data sets, assign and update datasets


```python
ds = f.create_dataset(
    "array2", 
    data=np.random.randint(10, size=10))
ds.value
```




    array([0, 2, 2, 4, 7, 3, 7, 2, 4, 1])




```python
ds = f.create_dataset(
    "/experiment2/simulation/data1", 
    shape=(5, 5), 
    fillvalue=-1)
ds.value
```




    array([[-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.]], dtype=float32)




```python
ds = f.create_dataset(
    "/experiment1/simulation/data1", 
    shape=(5000, 5000, 5000),
    fillvalue=0, 
    compression='gzip') # HDF5 = smart compression
ds
```




    <HDF5 dataset "data1": shape (5000, 5000, 5000), type "<f4">




```python
ds[:, 0, 0]  = np.random.rand(5000)
ds[1, :, 0] += np.random.rand(5000)
```


```python
ds[:2, :5, 0]
```




    array([[0.6939344 , 0.        , 0.        , 0.        , 0.        ],
           [1.4819994 , 0.01639538, 0.54387355, 0.11130908, 0.9928771 ]],
          dtype=float32)




```python
# if you need a reminder of the default value of a dataset:
ds.fillvalue
```




    0.0




```python
f["experiment1"].visititems(
    lambda name, 
    value: print(name, value))
```

    simulation <HDF5 group "/experiment1/simulation" (1 members)>
    simulation/data1 <HDF5 dataset "data1": shape (5000, 5000, 5000), type "<f4">



```python
f.flush()
f.filename
```




    'ch18-data.h5'




```python
# HDF5 = smart about file compression. Very larget dataset --> relatively small file size
!ls -lh ch18-data.h5
```

    -rw-rw-r-- 1 bjpcjp bjpcjp 357K May 19 17:47 ch18-data.h5



```python
# Deleting items from HDF5 file:
del f["/experiment1/simulation/data1"]
```


```python
# data1 should now be gone
f["experiment1"].visititems(
    lambda name, 
    value: print(name, value))
```

    simulation <HDF5 group "/experiment1/simulation" (0 members)>



```python
f.close()
```

### HDF5 Atributes

* Attributes make HDF5 great for annotating data & self-describing data (metadata).


```python
f = h5py.File("ch18-data.h5")
f.attrs
```




    <Attributes of HDF5 object at 140416821333160>




```python
# create an attribute
f.attrs["desc"] = "Result sets from experiments and simulations"
```


```python
f["experiment1"].attrs["date"] = "2015-1-1"
f["experiment2"].attrs["date"] = "2015-1-2"

f["experiment2/simulation/data1"].attrs["k"] = 1.5
f["experiment2/simulation/data1"].attrs["T"] = 1000
```


```python
list(f["experiment1"].attrs.keys())
```




    ['date']




```python
list(f["experiment2/simulation/data1"].attrs.items())
```




    [('k', 1.5), ('T', 1000)]




```python
# Existence testing:
"T" in f["experiment2/simulation/data1"].attrs
```




    True




```python
# Deleting existing attributes:
del f["experiment2/simulation/data1"].attrs["T"]
```


```python
"T" in f["experiment2/simulation/data1"].attrs
```




    False




```python
f["experiment2/simulation/data1"].attrs["t"] = np.array([1, 2, 3])
```


```python
f["experiment2/simulation/data1"].attrs["t"]
```




    array([1, 2, 3])




```python
f.close()
```

### pytables
* alternate HDF5 interface


```python
df = pd.read_csv(
    "ch18-playerstats-2013-2014-top30.csv", skiprows=1)

df = df.set_index("Rank")
```


```python
df[["Player", "Pos", "GP", "P", "G", "A", "S%", "Shift/GP"]].head(5)
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
      <th>Player</th>
      <th>Pos</th>
      <th>GP</th>
      <th>P</th>
      <th>G</th>
      <th>A</th>
      <th>S%</th>
      <th>Shift/GP</th>
    </tr>
    <tr>
      <th>Rank</th>
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
      <td>Sidney Crosby</td>
      <td>C</td>
      <td>80</td>
      <td>104</td>
      <td>36</td>
      <td>68</td>
      <td>13.9</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ryan Getzlaf</td>
      <td>C</td>
      <td>77</td>
      <td>87</td>
      <td>31</td>
      <td>56</td>
      <td>15.2</td>
      <td>25.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Claude Giroux</td>
      <td>C</td>
      <td>82</td>
      <td>86</td>
      <td>28</td>
      <td>58</td>
      <td>12.6</td>
      <td>25.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tyler Seguin</td>
      <td>C</td>
      <td>80</td>
      <td>84</td>
      <td>37</td>
      <td>47</td>
      <td>12.6</td>
      <td>23.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Corey Perry</td>
      <td>R</td>
      <td>81</td>
      <td>82</td>
      <td>43</td>
      <td>39</td>
      <td>15.4</td>
      <td>23.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create new PyTables HDF5 file
f = tables.open_file(
    "ch18-playerstats-2013-2014.h5", mode="w")
```


```python
# create HDF5 groups
grp = f.create_group(
    "/", 
    "season_2013_2014", 
    title="NHL player statistics for the 2013/2014 season")
grp
```




    /season_2013_2014 (Group) 'NHL player statistics for the 2013/2014 season'
      children := []




```python
# Unlike h5py, PyTables file objects do not represent root groups in the HDF5 file.
# Use the root attribute to access it instead.
f.root
```




    / (RootGroup) ''
      children := ['season_2013_2014' (Group)]




```python
# PyTables makes it easy to create mixed column types.

class PlayerStat(tables.IsDescription):
    player                 = tables.StringCol(20, dflt="")
    position               = tables.StringCol(1, dflt="C")
    games_played           = tables.UInt8Col(dflt=0)
    points                 = tables.UInt16Col(dflt=0)
    goals                  = tables.UInt16Col(dflt=0)
    assists                = tables.UInt16Col(dflt=0)
    shooting_percentage    = tables.Float64Col(dflt=0.0)
    shifts_per_game_played = tables.Float64Col(dflt=0.0) 
```


```python
top30_table = f.create_table(
    grp, 'top30', PlayerStat, "Top 30 point leaders")
```


```python
playerstat = top30_table.row
type(playerstat)
```




    tables.tableextension.Row




```python
# to insert data into table, use row attribute of table object
# when row object is initialized, use append to insert data.

for index, row_series in df.iterrows():
    playerstat["player"]                = row_series["Player"]    
    playerstat["position"]               = row_series["Pos"]    
    playerstat["games_played"]           = row_series["GP"]    
    playerstat["points"]                 = row_series["P"]    
    playerstat["goals"]                  = row_series["G"]
    playerstat["assists"]                = row_series["A"] 
    playerstat["shooting_percentage"]    = row_series["S%"]
    playerstat["shifts_per_game_played"] = row_series["Shift/GP"]
    playerstat.append()
```


```python
# flush forces a file write
top30_table.flush()
```


```python
# access table data using cols attribute
top30_table.cols.player[:5]
```




    array([b'Sidney Crosby', b'Ryan Getzlaf', b'Claude Giroux',
           b'Tyler Seguin', b'Corey Perry'], dtype='|S20')




```python
top30_table.cols.points[:5]
```




    array([104,  87,  86,  84,  82], dtype=uint16)




```python
# Use iterrows to create an iterator for row-wise data access.
def print_playerstat(row):
    print("%20s\t%s\t%s\t%s" %
          (row["player"].decode('UTF-8'), row["points"], row["goals"], row["assists"]))
```


```python
for row in top30_table.iterrows():
    print_playerstat(row)
```

           Sidney Crosby	104	36	68
            Ryan Getzlaf	87	31	56
           Claude Giroux	86	28	58
            Tyler Seguin	84	37	47
             Corey Perry	82	43	39
             Phil Kessel	80	37	43
             Taylor Hall	80	27	53
           Alex Ovechkin	79	51	28
            Joe Pavelski	79	41	38
              Jamie Benn	79	34	45
       Nicklas Backstrom	79	18	61
           Patrick Sharp	78	34	44
            Joe Thornton	76	11	65
           Erik Karlsson	74	20	54
           Evgeni Malkin	72	23	49
         Patrick Marleau	70	33	37
            Anze Kopitar	70	29	41
            Matt Duchene	70	23	47
        Martin St. Louis	69	30	39
            Patrick Kane	69	29	40
           Blake Wheeler	69	28	41
             Kyle Okposo	69	27	42
            David Krejci	69	19	50
            Chris Kunitz	68	35	33
          Jonathan Toews	68	28	40
            Thomas Vanek	68	27	41
            Jaromir Jagr	67	24	43
            John Tavares	66	24	42
            Jason Spezza	66	23	43
           Jordan Eberle	65	28	37



```python
# PyTables support SQL-like queries.
for row in top30_table.where("(points > 75) & (points <= 80)"):
    print_playerstat(row)
```

             Phil Kessel	80	37	43
             Taylor Hall	80	27	53
           Alex Ovechkin	79	51	28
            Joe Pavelski	79	41	38
              Jamie Benn	79	34	45
       Nicklas Backstrom	79	18	61
           Patrick Sharp	78	34	44
            Joe Thornton	76	11	65



```python
# PyTables queries using multiple column conditions:
for row in top30_table.where("(goals > 40) & (points < 80)"):
    print_playerstat(row)
```

           Alex Ovechkin	79	51	28
            Joe Pavelski	79	41	38



```python
# inspect HDF5 file structure
f
```




    File(filename=ch18-playerstats-2013-2014.h5, title='', mode='w', root_uep='/', filters=Filters(complevel=0, shuffle=False, bitshuffle=False, fletcher32=False, least_significant_digit=None))
    / (RootGroup) ''
    /season_2013_2014 (Group) 'NHL player statistics for the 2013/2014 season'
    /season_2013_2014/top30 (Table(30,)) 'Top 30 point leaders'
      description := {
      "assists": UInt16Col(shape=(), dflt=0, pos=0),
      "games_played": UInt8Col(shape=(), dflt=0, pos=1),
      "goals": UInt16Col(shape=(), dflt=0, pos=2),
      "player": StringCol(itemsize=20, shape=(), dflt=b'', pos=3),
      "points": UInt16Col(shape=(), dflt=0, pos=4),
      "position": StringCol(itemsize=1, shape=(), dflt=b'C', pos=5),
      "shifts_per_game_played": Float64Col(shape=(), dflt=0.0, pos=6),
      "shooting_percentage": Float64Col(shape=(), dflt=0.0, pos=7)}
      byteorder := 'little'
      chunkshape := (1489,)




```python
# done? let's flush the buffers, force a write & close the file.
f.flush()
f.close()
```

### Pandas hdfstore
* 3rd method to use HDF5 files - using HDFStore object in Pandas
* HDFStore object can be used as a dictionary for Pandas dataframes.


```python
import pandas as pd
```


```python
store = pd.HDFStore('store.h5')
```


```python
df = pd.DataFrame(np.random.rand(5,5))
store["df1"] = df
```


```python
df = pd.read_csv("ch18-playerstats-2013-2014-top30.csv", skiprows=1)
store["df2"] = df
```


```python
# What's in the HDFstore object?
store.keys()
```




    ['/df1', '/df2']




```python
# test for existence
'df2' in store
```




    True




```python
# retrieve an object
df = store["df1"]
```


```python
# access underlying HDF5 handle
store.root
```




    / (RootGroup) ''
      children := ['df1' (Group), 'df2' (Group)]




```python
store.close()
```


```python
# HDF5 is a std file format. We can open a file & see how data is arranged.
f = h5py.File("store.h5")
```


```python
f.visititems(
    lambda x, y: 
    print(x, "\t" * int(3 - len(str(x))//8), y))
```

    df1 			 <HDF5 group "/df1" (4 members)>
    df1/axis0 		 <HDF5 dataset "axis0": shape (5,), type "<i8">
    df1/axis1 		 <HDF5 dataset "axis1": shape (5,), type "<i8">
    df1/block0_items 	 <HDF5 dataset "block0_items": shape (5,), type "<i8">
    df1/block0_values 	 <HDF5 dataset "block0_values": shape (5, 5), type "<f8">
    df2 			 <HDF5 group "/df2" (8 members)>
    df2/axis0 		 <HDF5 dataset "axis0": shape (21,), type "|S8">
    df2/axis1 		 <HDF5 dataset "axis1": shape (30,), type "<i8">
    df2/block0_items 	 <HDF5 dataset "block0_items": shape (3,), type "|S8">
    df2/block0_values 	 <HDF5 dataset "block0_values": shape (30, 3), type "<f8">
    df2/block1_items 	 <HDF5 dataset "block1_items": shape (14,), type "|S4">
    df2/block1_values 	 <HDF5 dataset "block1_values": shape (30, 14), type "<i8">
    df2/block2_items 	 <HDF5 dataset "block2_items": shape (4,), type "|S6">
    df2/block2_values 	 <HDF5 dataset "block2_values": shape (1,), type "|O">



```python
# HDF5Store objects store dataframes in distinct groups.
# Each dataframe is split into heterogeneous "blocks" with columns grouped by data type
# Column names & values are stored in separate HDF5 datasets.

f["/df2/block0_items"].value          
```




    array([b'S%', b'Shift/GP', b'FO%'], dtype='|S8')




```python
f["/df2/block0_values"][:3]
```




    array([[13.9, 24. , 52.5],
           [15.2, 25.2, 49. ],
           [12.6, 25.1, 52.9]])




```python
f["/df2/block1_items"].value  
```




    array([b'Rank', b'GP', b'G', b'A', b'P', b'+/-', b'PIM', b'PPG', b'PPP',
           b'SHG', b'SHP', b'GW', b'OT', b'S'], dtype='|S4')




```python
f["/df2/block1_values"][:3, :5]
```




    array([[  1,  80,  36,  68, 104],
           [  2,  77,  31,  56,  87],
           [  3,  82,  28,  58,  86]])



### JSON
* Human-readable, lightweight plain-text format
* Ideal for storing lists & dictionaries - no tabular data restrictions


```python
# storing Python lists as JSON strings
data      = ["string", 1.0, 2, None]
data_json = json.dumps(data)
data_json
```




    '["string", 1.0, 2, null]'




```python
# parsing JSON string to a Python object
data2 = json.loads(data_json)
data2
```




    ['string', 1.0, 2, None]




```python
# Storing Python dictionaries as JSON strings
data = {"one": 1, "two": 2.0, "three": "three"}
data_json = json.dumps(data)
print(data_json)
```

    {"one": 1, "two": 2.0, "three": "three"}



```python
# Parsing JSON string back to Python object
data = json.loads(data_json)
data["two"], data["three"]
```




    (2.0, 'three')




```python
# JSON can handle variable-size elements
data = {"one": [1], 
        "two": [1, 2], 
        "three": [1, 2, 3]}
```


```python
# indent=True obtains indented JSON code = easier to read.
data_json = json.dumps(data, indent=True)
print(data_json)
```

    {
     "one": [
      1
     ],
     "two": [
      1,
      2
     ],
     "three": [
      1,
      2,
      3
     ]
    }



```python
# another complex data structure, saved to a JSON file
data = {"one": [1], 
        "two": {"one": 1, "two": 2}, 
        "three": [(1,), (1, 2), (1, 2, 3)],
        "four": "a text string"}
```


```python
with open("data.json", "w") as f:
    json.dump(data, f)
```


```python
!cat data.json
```

    {"one": [1], "two": {"one": 1, "two": 2}, "three": [[1], [1, 2], [1, 2, 3]], "four": "a text string"}


```python
# read back into a Python object
with open("data.json", "r") as f:
    data_from_file = json.load(f)
```


```python
data_from_file["two"]
```




    {'one': 1, 'two': 2}




```python
data_from_file["three"]
```




    [[1], [1, 2], [1, 2, 3]]




```python
# revisit Tokyo metro dataset (JSON format):
!head -n 20 tokyo-metro.json
```

    {
        "C": {
            "color": "#149848", 
            "transfers": [
                [
                    "C3", 
                    "F15"
                ], 
                [
                    "C4", 
                    "Z2"
                ], 
                [
                    "C4", 
                    "G2"
                ], 
                [
                    "C7", 
                    "M14"
                ], 



```python
with open("tokyo-metro.json", "r") as f:
    data = json.load(f)
```


```python
# dictionary with a key for each metro line
data.keys()
```




    dict_keys(['C', 'G', 'F', 'H', 'M', 'N', 'T', 'Y', 'Z'])




```python
data["C"].keys()
```




    dict_keys(['color', 'transfers', 'travel_times'])




```python
data["C"]["color"]
```




    '#149848'




```python
data["C"]["transfers"]
```




    [['C3', 'F15'],
     ['C4', 'Z2'],
     ['C4', 'G2'],
     ['C7', 'M14'],
     ['C7', 'N6'],
     ['C7', 'G6'],
     ['C8', 'M15'],
     ['C8', 'H6'],
     ['C9', 'H7'],
     ['C9', 'Y18'],
     ['C11', 'T9'],
     ['C11', 'M18'],
     ['C11', 'Z8'],
     ['C12', 'M19'],
     ['C18', 'H21']]




```python
# now we can iterate & filter items with Python list syntax
# below: select connected nodes in graph, on C line, with travel time = 1 minute.

[(s, e, tt) for s, e, tt in data["C"]["travel_times"] if tt == 1]
```




    [('C3', 'C4', 1), ('C7', 'C8', 1), ('C9', 'C10', 1)]



### Serialization

* JSON files aren't space efficient, and can represent only limited set of datatypes.
* Two alternatives: __msgpack__ library, and Python's __pickle__ module.


```python
!ls -lh tokyo-metro.json
```

    -rw-rw-r-- 1 bjpcjp bjpcjp 27K May 19 16:47 tokyo-metro.json



```python
# packing the JSON "data" file (above) --> considerably smaller file
data_pack = msgpack.packb(data)
type(data_pack), len(data_pack)
```




    (bytes, 3021)




```python
with open("tokyo-metro.msgpack", "wb") as f:
    f.write(data_pack)
```


```python
!ls -lh tokyo-metro.msgpack
```

    -rw-rw-r-- 1 bjpcjp bjpcjp 3.0K May 19 18:46 tokyo-metro.msgpack



```python
# unpack the msgpack file, back into JSON:
with open("tokyo-metro.msgpack", "rb") as f:
    data_msgpack = f.read()
    data = msgpack.unpackb(data_msgpack)
```


```python
list(data.keys())
```




    [b'C', b'G', b'F', b'H', b'M', b'N', b'T', b'Y', b'Z']



### Serialization - pickle
* Advantage: almost any type of Python object can be serialized.
* Disadvantage: Pickles can't be read by any non-Python code. Also, not recommended for long-term storage due to version thrashing.


```python
with open("tokyo-metro.pickle", "wb") as f:
    cPickle.dump(data, f)
```


```python
!ls -lh tokyo-metro.pickle
```

    -rw-rw-r-- 1 bjpcjp bjpcjp 8.6K May 19 18:53 tokyo-metro.pickle



```python
with open("tokyo-metro.pickle", "rb") as f:
    data = pickle.load(f)
```


```python
data.keys()
```




    dict_keys([b'C', b'G', b'F', b'H', b'M', b'N', b'T', b'Y', b'Z'])


