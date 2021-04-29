#!/usr/bin/env python
# coding: utf-8

# ### Chapter 1: Computing with Python

# ### Overview: a typical Python-based scientific computing stack.
# ![software stack](pics/software-stack-overview.png)
# 
# ### Resources:
# - [Intel MKL (Math Kernel Library)](https://software.intel.com/en-us/intel-mkl)
# - [openBLAS](https://www.openblas.net)
# - [ATLAS](https://math-atlas.sourceforge.net)
# - [SciPy](http://www.scipy.org)
# - [Python Numeric & Scientific topics](http://wiki.python.org/moin/NumericAndScientific)

# ### Interpreter
# - The easist way to execute Python code: run the program directly.
# - Use Jupyter magic command to write Python source file to disk:

# In[1]:


get_ipython().run_cell_magic('writefile', 'hello.py', 'print("Hello from Python!")')


# * Use the ! system shell command (included in the Python Jupyter kernel) to interactively run Python with hello.py as its argument.

# In[2]:


get_ipython().system('python hello.py')


# In[3]:


get_ipython().system('python --version')


# ### Input and output caching
# 
# * Input & output history can be accessed using __In__ (a list) & __Out__ (a dictionary). Both can be indexed with a cell number. 

# In[4]:


3*3
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


# ### Autocompletion
# 
# * The __Tab__ key activates autocompletion (displays list of symbol names that are valid completions of what has been typed thus far.)

# In[11]:


import os


# * Results of typing "os.w", followed by \t:
# 
# ![autocompletion](pics/autocompletion.png)

# ### Documentation
# 
# * "Docstrings" provide a built-in reference manual for most Python modules. Display the docstring by appending a Python object with "?".

# In[12]:


import math
get_ipython().run_line_magic('pinfo', 'math.cos')


# ### Interaction with System Shell
# 
# * Anything after ! is evaluated using the system shell, such as bash.
# * (I use Ubuntu Linux as my laptop OS. Your Windows equivalents will vary.)

# In[13]:


get_ipython().system('touch file1.py file2.py file3.py')
get_ipython().system('ls file*')


# In[14]:


# output of a system shell command can be captured in a Python variable
files = get_ipython().getoutput('ls file*')
print(len(files))
print(files)


# In[15]:


# pass Python variable values to shell commands
# by prefixing the variable name with $.
file = "file1.py"
get_ipython().system('ls -l $file')


# ### IPython Extensions
# 
# * Commands start with one or two "%" characters. A single % is used for single-line commands; dual %% is used for cells (multiple lines).
# 
# * `%lsmagic` returns a list of available commands.

# In[16]:


get_ipython().run_line_magic('lsmagic', '')


# ### Running scripts
# 
# - `%run` executes an external Python source file within an interactive IPython session.

# In[17]:


get_ipython().run_cell_magic('writefile', 'fib.py', '\ndef fib(N): \n    """ \n    Return a list of the first N Fibonacci numbers.\n    """ \n    f0, f1 = 0, 1\n    f = [1] * N\n    for n in range(1, N):\n        f[n] = f0 + f1\n        f0, f1 = f1, f[n]\n\n    return f\n\nprint(fib(10))')


# In[18]:


get_ipython().system('python fib.py')


# In[19]:


get_ipython().run_line_magic('run', 'fib.py')


# In[20]:


fib(6)


# ## Listing all defined symbols
# 
# * `%who` lists all defined symbols
# * `%whos` provides more detailed info.

# In[21]:


get_ipython().run_line_magic('who', '')


# In[22]:


get_ipython().run_line_magic('whos', '')


# ## Debugger
# 
# * Use `%debug` to step directly into the Python debugger.

# In[23]:


# fib function fails - can't use floating point numbers.
try:
    fib(1.0)
except TypeError:
    print("nope. can't do that.")


# In[24]:


#%debug


# ### Resetting the Python namespace
# - Ensures a program is run in a pristine environment, uncluttered by existing variables and functions. Although it is necessary to reimport modules, it is important to know that even if the modules have changed since the last
# import, a new import after a %reset will not import the new module but rather reenable a cached version from the previous import. 
# 
# - When developing Python modules, this is usually not the desired behavior. In that case, a reimport of a previously imported (and since updated) module can often be achieved by using the reload function from `IPython.lib.deepreload`. However, this method does not always work, as
# some libraries run code at import time that is only intended to run once. In this case, the only option might be to terminate and restart the IPython interpreter.

# In[26]:


get_ipython().run_line_magic('reset', '')


# ## Timing and profiling code
# 
# * `%timeit` and `%time` provide simple benchmarking utilities.

# In[25]:


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


# In[26]:


# timeit does not return expression's resulting value.
get_ipython().run_line_magic('timeit', 'fib(50)')


# In[27]:


# %time only runs once. less accurate estimate.
result = get_ipython().run_line_magic('time', 'fib(100)')


# In[28]:


len(result)


# * The __cProfile__ module provides __%prun__ (for statements) and __%run__ (for external scripts) profiling commands.

# In[29]:


import numpy as np

def random_walker_max_distance(M, N):
    """
    Simulate N random walkers taking M steps
    Return the largest distance from the starting point.
    """
    trajectories = [np.random.randn(M).cumsum() 
                    for _ in range(N)]
    return np.max(np.abs(trajectories))


# In[30]:


# returns call counts, runtime & cume runtime for
# each function.
get_ipython().run_line_magic('prun', 'random_walker_max_distance(400, 10000)')


# ### Jupyter: External image rendering

# In[31]:


from IPython.display import display, Image, HTML, Math


# In[32]:


Image(url='http://python.org/images/python-logo.gif')


# ### Jupyter: HTML rendering

# In[33]:


import scipy, numpy, matplotlib
modules = [numpy, matplotlib, scipy]

row = "<tr><td>%s</td><td>%s</td></tr>"
rows = "\n".join(
    [row % 
     (module.__name__, module.__version__) 
     for module in modules])
table = "<table><tr><th>Library</th><th>Version</th></tr> %s </table>" % rows


# In[34]:


HTML(table)


# In[35]:


# another method
class HTMLdisplayer(object):
    def __init__(self,code):
        self.code = code
    def _repr_html_(self):
        return self.code
    
HTMLdisplayer(table)


# ### Jupyter: Formula rendering using Latex

# In[36]:


Math(r'\hat{H} = -\frac{1}{2}\epsilon \hat{\sigma}_z-\frac{1}{2}\delta \hat{\sigma}_x')


# ### Jupyter: UI Widgets
# 
# ** Needs debugging: slider widget doesn't appear. **

# In[37]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def f(mu):
    X = stats.norm(loc=mu, scale=np.sqrt(mu))
    N = stats.poisson(mu)
    x = np.linspace(0, X.ppf(0.999))
    n = np.arange(0, x[-1])
    
    fig, ax = plt.subplots()
    ax.plot(x, X.pdf(x), color='black', lw=2, label="Normal($\mu=%d, \sigma^2=%d$)" % (mu,mu))
    ax.bar(n, N.pmf(n), align='edge', label=r"Poisson($\lambda=%d$)" % mu)
    ax.set_ylim(0, X.pdf(x).max() * 1.25)
    ax.legend(loc=2, ncol=2)
    plt.close(fig)
    return fig


# In[38]:


from ipywidgets import interact
import ipywidgets as widgets


# In[39]:


interact(f, mu=widgets.FloatSlider(min=1.0, max=20.0, step=1.0));


# ### nbconvert to HTML

# In[40]:


get_ipython().system('jupyter nbconvert --to html ch01-intro.ipynb')


# ### nbconvert to PDF
# * [Requires a LaTeX environment](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex) to be installed.
# * On this system (Ubuntu Linux): ```sudo apt-get install texlive-xetex```

# In[41]:


get_ipython().system('jupyter nbconvert --to pdf ch01-intro.ipynb;')


# ## nbconvert to pure Python source code

# In[70]:


get_ipython().system('jupyter nbconvert ch01-intro.ipynb --to python')


# In[71]:


get_ipython().system('ls ch01*')


# In[ ]:




