
# coding: utf-8

# # Chapter 1: Computing with Python

# ### Overview: a typical Python-based scientific computing stack.
# ![software stack](pics/software-stack-overview.png)

# ## Interpreter
# - The easist way to execute Python code: run the program directly.
# - Use Jupyter magic command to write Python source file to disk:

# In[1]:


get_ipython().run_cell_magic('writefile', 'hello.py', 'print("Hello from Python!")')


# * Use the ! system shell command (included in the Python Jupyter kernel) to interactively run Python with hello.py as its argument.

# In[1]:


get_ipython().system('python hello.py')


# In[2]:


get_ipython().system('python --version')


# ## Input and output caching
# 
# * Input & output history can be accessed using __In__ (a list) & __Out__ (a dictionary). Both can be indexed with a cell number. 

# In[3]:


3 * 3


# In[4]:


In[1]


# * A single underscore = the most recent output; 
# * A double underscore = the _next_ most recent output.

# In[5]:


1+1


# In[6]:


2+2


# In[7]:


_, __


# In[8]:


# In = a list
In


# In[9]:


# Out = a dictionary
Out


# In[10]:


# Suppress output results by ending statement with a semicolon
1+2;


# ## Autocompletion
# 
# * The __Tab__ key activates autocompletion (displays list of symbol names that are valid completions of what has been typed thus far.)

# In[11]:


import os


# * Results of typing "os.w", followed by \t:
# 
# ![autocompletion](pics/autocompletion.png)

# ## Documentation
# 
# * "Docstrings" provide a built-in reference manual for most Python modules. Display the docstring by appending a Python object with "?".

# In[12]:


import math


# In[13]:


get_ipython().run_line_magic('pinfo', 'math.cos')


# ## Interaction with System Shell
# 
# * Anything after ! is evaluated using the system shell, such as bash.

# In[14]:


get_ipython().system('touch file1.py file2.py file3.py')


# In[15]:


get_ipython().system('ls file*')


# In[16]:


# output of a system shell command
# can be captured in a Python variable
files = get_ipython().getoutput('ls file*')


# In[17]:


len(files)


# In[18]:


files


# In[19]:


# pass Python variable values to shell commands
# by prefixing the variable name with $.
file = "file1.py"


# In[20]:


get_ipython().system('ls -l $file')


# ## IPython Extensions
# 
# * Commands start with one or two "%" characters. A single % is used for single-line commands; dual %% is used for cells (multiple lines).
# 
# * %lsmagic returns a list of available commands.

# In[21]:


get_ipython().run_line_magic('lsmagic', '')


# ## Running scripts

# In[22]:


get_ipython().run_cell_magic('writefile', 'fib.py', '\ndef fib(N): \n    """ \n    Return a list of the first N Fibonacci numbers.\n    """ \n    f0, f1 = 0, 1\n    f = [1] * N\n    for n in range(1, N):\n        f[n] = f0 + f1\n        f0, f1 = f1, f[n]\n\n    return f\n\nprint(fib(10))')


# In[23]:


get_ipython().system('python fib.py')


# In[24]:


get_ipython().run_line_magic('run', 'fib.py')


# In[25]:


fib(6)


# ## Listing all defined symbols
# 
# * __%who__ lists all defined symbols
# * __%whos__ provides more detailed info.

# In[26]:


get_ipython().run_line_magic('who', '')


# In[27]:


get_ipython().run_line_magic('whos', '')


# ## Debugger
# 
# * Use __%debug__ to step directly into the Python debugger.

# In[30]:


# fib function fails - can't use floating point numbers.
fib(1.0)


# In[29]:


get_ipython().run_line_magic('debug', '')


# ## Resetting the Python namespace

# In[32]:


get_ipython().run_line_magic('reset', '')


# ## Timing and profiling code
# 
# * __%timeit__ and __%time__ provide simple benchmarking utilities.

# In[34]:


# first, re-define fibonacci code used above.
def fib(N): 
    """ 
    Return a list of the first N Fibonacci numbers.
    """ 
    f0, f1 = 0, 1
    f = [1] * N
    for n in range(1, N):
        f[n] = f0 + f1
        f0, f1 = f1, f[n]

    return f


# In[35]:


get_ipython().run_line_magic('timeit', 'fib(50)')


# In[36]:


# %time only runs once. less accurate estimate.
result = get_ipython().run_line_magic('time', 'fib(100)')


# In[42]:


len(result)


# * The __cProfile__ module provides __%prun__ (for statements) and __%run__ (for external scripts) profiling commands.

# In[37]:


import numpy as np

def random_walker_max_distance(M, N):
    """
    Simulate N random walkers taking M steps
    Return the largest distance from the starting point.
    """
    trajectories = [np.random.randn(M).cumsum() 
                    for _ in range(N)]
    return np.max(np.abs(trajectories))


# In[39]:


# returns call counts, runtime & cume runtime for
# each function.
get_ipython().run_line_magic('prun', 'random_walker_max_distance(400, 10000)')


# ## nbconvert to HTML file

# In[40]:


get_ipython().system('jupyter nbconvert --to html ch01-intro.ipynb')


# ## nbconvert to PDF file
# * [Requires a LaTeX environment](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex) to be installed.
# * On this system (Ubuntu Linux): ```sudo apt-get install texlive-xetex```

# In[2]:


get_ipython().system('jupyter nbconvert --to pdf ch01-intro.ipynb;')


# ## nbconvert to pure Python source code

# In[49]:


get_ipython().system('jupyter nbconvert ch01-intro.ipynb --to python')


# In[50]:


get_ipython().system('ls ch01*')

