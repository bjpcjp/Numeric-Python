
# coding: utf-8

# # Chapter 1: Computing with Python

# ## Interpreter

# In[1]:

get_ipython().run_cell_magic('writefile', 'hello.py', 'print("Hello from Python!")')


# In[2]:

get_ipython().system('python hello.py')


# In[3]:

get_ipython().system('python --version')


# ## Input and output caching

# In[4]:

3 * 3


# In[5]:

In[1]


# In[6]:

Out[1]


# In[7]:

In


# In[8]:

Out


# In[9]:

1+2


# In[10]:

1+2;


# In[11]:

x = 1


# In[12]:

x = 2; x


# ## Documentation

# In[13]:

import os


# In[14]:

# try os.w<TAB>


# In[15]:

import math


# In[ ]:

get_ipython().magic('pinfo math.cos')


# ## Interaction with System Shell

# In[16]:

get_ipython().system('touch file1.py file2.py file3.py')


# In[17]:

get_ipython().system('ls file*')


# In[18]:

files = get_ipython().getoutput('ls file*')


# In[19]:

len(files)


# In[20]:

files


# In[21]:

file = "file1.py"


# In[22]:

get_ipython().system('ls -l $file')


# ## Running scripts from the IPython console

# In[23]:

get_ipython().run_cell_magic('writefile', 'fib.py', '\ndef fib(N): \n    """ \n    Return a list of the first N Fibonacci numbers.\n    """ \n    f0, f1 = 0, 1\n    f = [1] * N\n    for n in range(1, N):\n        f[n] = f0 + f1\n        f0, f1 = f1, f[n]\n\n    return f\n\nprint(fib(10))')


# In[24]:

get_ipython().system('python fib.py')


# In[25]:

get_ipython().magic('run fib.py')


# In[26]:

fib(6)


# ## Debugger

# In[27]:

fib(1)


# In[ ]:

get_ipython().magic('debug')


# ## Timing and profiling code

# In[28]:

get_ipython().magic('timeit fib(100)')


# In[29]:

result = get_ipython().magic('time fib(100)')


# In[ ]:

len(result)


# In[ ]:

import numpy as np

def random_walker_max_distance(M, N):
    """
    Simulate N random walkers taking M steps, and return the largest distance
    from the starting point achieved by any of the random walkers.
    """
    trajectories = [np.random.randn(M).cumsum() for _ in range(N)]
    return np.max(np.abs(trajectories))


# In[ ]:

get_ipython().magic('prun random_walker_max_distance(400, 10000)')


# ## IPython nbconvert

# In[ ]:

get_ipython().system('ipython nbconvert --to html ch01-code-listing.ipynb')


# In[ ]:

get_ipython().system('ipython nbconvert --to pdf ch01-code-listing.ipynb')


# In[ ]:

get_ipython().run_cell_magic('writefile', 'custom_template.tplx', "((*- extends 'article.tplx' -*))\n\n((* block title *)) \\title{Document title} ((* endblock title *))\n((* block author *)) \\author{Author's Name} ((* endblock author *))")


# In[ ]:

get_ipython().system('ipython nbconvert ch01-code-listing.ipynb --to pdf --template custom_template.tplx')


# In[ ]:

get_ipython().system('ipython nbconvert ch01-code-listing.ipynb --to python')


# # Versions

# In[ ]:

get_ipython().magic('reload_ext version_information')
get_ipython().magic('version_information numpy')

