
# coding: utf-8

# # Chapter 1: Computing with Python

# ### Overiew: a typical Python-based scientific computing stack.
# ![software stack](pics/software-stack-overview.png)

# ## Interpreter

# In[1]:


get_ipython().run_cell_magic('writefile', 'hello.py', 'print("Hello from Python!")')


# In[2]:


get_ipython().system('python hello.py')


# In[3]:


get_ipython().system('python --version')


# ## Input and output caching
# 
# Input prompt shown as __In [1]__; corresponding output shown as __Out [1]__.

# In[4]:


3 * 3


# In[5]:


In[1]


# A single underscore = the most recent output; a double underscore = the _next_ most recent output.

# In[6]:


1+1


# In[7]:


2+2


# In[8]:


_, __


# __In__ is an array:

# In[9]:


In


# __Out__ is a dictionary:

# In[10]:


Out


# Suppress output results by ending statement with a semicolon:

# In[11]:


1+2;


# ## Autocompletion
# 
# The __Tab__ key activates autocompletion (displays list of symbol names that are valid completions of what has been typed thus far.)

# In[12]:


import os


# Results of typing "os.w", followed by \t:
# 
# ![autocompletion](pics/autocompletion.png)

# ## Documentation
# 
# "Docstrings" provide a built-in reference manual for most Python modules. Display the docstring by appending a Python object with "?".

# In[13]:


import math


# In[14]:


get_ipython().run_line_magic('pinfo', 'math.cos')


# ## Interaction with System Shell
# 
# (In this case, Ubuntu Linux.)

# In[15]:


get_ipython().system('touch file1.py file2.py file3.py')


# In[16]:


get_ipython().system('ls file*')


# In[17]:


files = get_ipython().getoutput('ls file*')


# In[18]:


len(files)


# In[19]:


files


# In[20]:


file = "file1.py"


# In[21]:


get_ipython().system('ls -l $file')


# ## IPython Extensions
# 
# Commands start with one or two "%" characters. A single % is used for single-line commands; dual %% is used for cells (multiple lines).
# 
# %lsmagic returns a list of available commands.

# In[25]:


get_ipython().run_line_magic('pinfo', '%lsmagic')


# ## Running scripts using %run, or by using !python

# In[26]:


get_ipython().run_cell_magic('writefile', 'fib.py', '\ndef fib(N): \n    """ \n    Return a list of the first N Fibonacci numbers.\n    """ \n    f0, f1 = 0, 1\n    f = [1] * N\n    for n in range(1, N):\n        f[n] = f0 + f1\n        f0, f1 = f1, f[n]\n\n    return f\n\nprint(fib(10))')


# In[27]:


get_ipython().system('python fib.py')


# In[28]:


get_ipython().run_line_magic('run', 'fib.py')


# In[29]:


fib(6)


# __%who__ lists all defined symbols; __%whos__ provides more detailed info.

# In[30]:


get_ipython().run_line_magic('who', '')


# In[31]:


get_ipython().run_line_magic('whos', '')


# ## Debugger
# 
# Use __%debug__ to step directly into the Python debugger.

# In[37]:


fib(1.0)


# In[38]:


get_ipython().run_line_magic('debug', '')


# ## Timing and profiling code
# 
# __%timeit__ and __%time__ provide simple benchmarking utilities.

# In[39]:


get_ipython().run_line_magic('timeit', 'fib(50)')


# In[41]:


result = get_ipython().run_line_magic('time', 'fib(100)')


# In[42]:


len(result)


# The Python __cProfile__ module (which is standard) provides the __%prun__ (for statements) and __%run__ (for external scripts) profiling commands. Consider the following: 

# In[43]:


import numpy as np

def random_walker_max_distance(M, N):
    """
    Simulate N random walkers taking M steps
    Return the largest distance from the starting point.
    """
    trajectories = [np.random.randn(M).cumsum() for _ in range(N)]
    return np.max(np.abs(trajectories))


# In[44]:


get_ipython().run_line_magic('prun', 'random_walker_max_distance(400, 10000)')


# ## nbconvert

# In[45]:


get_ipython().system('jupyter nbconvert --to html ch01-intro.ipynb')


# ## nbconvert to PDF file
# 
# Re

# In[36]:


get_ipython().system('ipython nbconvert ch01-code-listing.ipynb --to python')

