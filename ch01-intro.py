
# coding: utf-8

# # Chapter 1: Computing with Python

# ### Overview: a typical Python-based scientific computing stack.
# ![software stack](pics/software-stack-overview.png)
# 
# * Resources:
# - [SciPy](http://www.scipy.org)
# - [Python Numeric & Scientific topics](http://wiki.python.org/moin/NumericAndScientific)

# ## Interpreter
# - The easist way to execute Python code: run the program directly.
# - Use Jupyter magic command to write Python source file to disk:

# In[1]:


get_ipython().run_cell_magic('writefile', 'hello.py', 'print("Hello from Python!")')


# * Use the ! system shell command (included in the Python Jupyter kernel) to interactively run Python with hello.py as its argument.

# In[2]:


get_ipython().system('python hello.py')


# In[3]:


get_ipython().system('python --version')


# ## Input and output caching
# 
# * Input & output history can be accessed using __In__ (a list) & __Out__ (a dictionary). Both can be indexed with a cell number. 

# In[4]:


3 * 3


# In[5]:


In[1]


# * A single underscore = the most recent output; 
# * A double underscore = the _next_ most recent output.

# In[6]:


1+1


# In[7]:


2+2


# In[8]:


_, __


# In[9]:


# In = a list
In


# In[10]:


# Out = a dictionary
Out


# In[11]:


# Suppress output results by ending statement with a semicolon
1+2;


# ## Autocompletion
# 
# * The __Tab__ key activates autocompletion (displays list of symbol names that are valid completions of what has been typed thus far.)

# In[12]:


import os


# * Results of typing "os.w", followed by \t:
# 
# ![autocompletion](pics/autocompletion.png)

# ## Documentation
# 
# * "Docstrings" provide a built-in reference manual for most Python modules. Display the docstring by appending a Python object with "?".

# In[13]:


import math


# In[14]:


get_ipython().run_line_magic('pinfo', 'math.cos')


# ## Interaction with System Shell
# 
# * Anything after ! is evaluated using the system shell, such as bash.
# * (I use Ubuntu Linux as my laptop OS. Your Windows equivalents will vary.)

# In[15]:


get_ipython().system('touch file1.py file2.py file3.py')


# In[16]:


get_ipython().system('ls file*')


# In[17]:


# output of a system shell command
# can be captured in a Python variable
files = get_ipython().getoutput('ls file*')


# In[18]:


len(files)


# In[19]:


files


# In[20]:


# pass Python variable values to shell commands
# by prefixing the variable name with $.
file = "file1.py"


# In[21]:


get_ipython().system('ls -l $file')


# ## IPython Extensions
# 
# * Commands start with one or two "%" characters. A single % is used for single-line commands; dual %% is used for cells (multiple lines).
# 
# * %lsmagic returns a list of available commands.

# In[23]:


get_ipython().run_line_magic('pinfo', '%lsmagic')


# ## Running scripts

# In[24]:


get_ipython().run_cell_magic('writefile', 'fib.py', '\ndef fib(N): \n    """ \n    Return a list of the first N Fibonacci numbers.\n    """ \n    f0, f1 = 0, 1\n    f = [1] * N\n    for n in range(1, N):\n        f[n] = f0 + f1\n        f0, f1 = f1, f[n]\n\n    return f\n\nprint(fib(10))')


# In[25]:


get_ipython().system('python fib.py')


# In[26]:


get_ipython().run_line_magic('run', 'fib.py')


# In[27]:


fib(6)


# ## Listing all defined symbols
# 
# * __%who__ lists all defined symbols
# * __%whos__ provides more detailed info.

# In[28]:


get_ipython().run_line_magic('who', '')


# In[29]:


get_ipython().run_line_magic('whos', '')


# ## Debugger
# 
# * Use __%debug__ to step directly into the Python debugger.

# In[30]:


# fib function fails - can't use floating point numbers.
fib(1.0)


# In[32]:


get_ipython().run_line_magic('debug', '')


# ## Resetting the Python namespace

# In[34]:


get_ipython().run_line_magic('reset', '')


# ## Timing and profiling code
# 
# * __%timeit__ and __%time__ provide simple benchmarking utilities.

# In[35]:


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


# In[36]:


get_ipython().run_line_magic('timeit', 'fib(50)')


# In[37]:


# %time only runs once. less accurate estimate.
result = get_ipython().run_line_magic('time', 'fib(100)')


# In[38]:


len(result)


# * The __cProfile__ module provides __%prun__ (for statements) and __%run__ (for external scripts) profiling commands.

# In[39]:


import numpy as np

def random_walker_max_distance(M, N):
    """
    Simulate N random walkers taking M steps
    Return the largest distance from the starting point.
    """
    trajectories = [np.random.randn(M).cumsum() 
                    for _ in range(N)]
    return np.max(np.abs(trajectories))


# In[40]:


# returns call counts, runtime & cume runtime for
# each function.
get_ipython().run_line_magic('prun', 'random_walker_max_distance(400, 10000)')


# ### Jupyter: External image rendering

# In[41]:


from IPython.display import display, Image, HTML, Math


# In[42]:


Image(url='http://python.org/images/python-logo.gif')


# ### Jupyter: HTML rendering

# In[47]:


import scipy, numpy, matplotlib
modules = [numpy, matplotlib, scipy]

row = "<tr><td>%s</td><td>%s</td></tr>"
rows = "\n".join(
    [row % 
     (module.__name__, module.__version__) 
     for module in modules])
table = "<table><tr><th>Library</th><th>Version</th></tr> %s </table>" % rows


# In[48]:


HTML(table)


# In[50]:


# another method
class HTMLdisplayer(object):
    def __init__(self,code):
        self.code = code
    def _repr_html_(self):
        return self.code
    
HTMLdisplayer(table)


# ### Jupyter: Formula rendering using Latex

# In[53]:


Math(r'\hat{H} = -\frac{1}{2}\epsilon \hat{\sigma}_z-\frac{1}{2}\delta \hat{\sigma}_x')


# ### Jupyter: UI Widgets
# 
# ** Needs debugging: slider widget doesn't appear. **

# In[54]:


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


# In[55]:


from ipywidgets import interact
import ipywidgets as widgets


# In[58]:


interact(f, mu=widgets.FloatSlider(min=1.0, max=20.0, step=1.0));


# ## nbconvert to HTML file

# In[59]:


get_ipython().system('jupyter nbconvert --to html ch01-intro.ipynb')


# ## nbconvert to PDF file
# * [Requires a LaTeX environment](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex) to be installed.
# * On this system (Ubuntu Linux): ```sudo apt-get install texlive-xetex```

# In[60]:


get_ipython().system('jupyter nbconvert --to pdf ch01-intro.ipynb;')


# ## nbconvert to pure Python source code

# In[4]:


get_ipython().system('jupyter nbconvert ch01-intro.ipynb --to python')


# In[5]:


get_ipython().system('ls ch01*')

