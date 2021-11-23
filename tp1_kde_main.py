
# coding: utf-8

# ## TP3 KDE Kernel Density Estimation
# * obligatory
# * individual work
# * The report (.pdf or jupyter notebook)  
#     - should start with a short introdution
#     - to explan the problem, 
#     - explan a bit the specific method that you are going to use
#     - should include a detailed description of your observations, e.g. comments on the forms of the density functions, the classification performance.

# The TP is divided in three parts: 
# * The first part concerns the definition of the appropriate functions for probability density estimation using kernels and the study of the effect of the h parameter on a simple artificial set. 
# * The second concerns the application of the functions written previously on the iris dataset. 
# * The third is to apply the density estimation to a classification problem.



from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.pyplot import show
from itertools import combinations

# make figures appear inline
# get_ipython().run_line_magic('matplotlib', 'inline')

# notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# ## Define training set
# Let your training set consist of four points

# In[ ]:


# Define training set
c = np.array([[1,1],[1,4],[3,2.5],[4,2.5]])
n = c.shape[0]


# ## Define test set
# Create a regular set of points which cover the plane $[0, 5] \times [0, 5]$  and stores them in testSet

# In[ ]:


# Define test set
min_X, max_X = 0, 5
min_Y, max_Y = 0, 5
intLength = 30
x = np.linspace(min_X, max_X, num=intLength)
y = np.linspace(min_Y, max_Y, num=intLength)
testSet = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
X,Y = np.meshgrid(x,y)


# In[ ]:


print('testSet')
print( testSet)
print('testSet shape: ', testSet.shape)


# ## Define of the appropriate functions for probability density estimation using kernels 
# 
# Open ```kde.py```, here you can find some examples you might need. you can either complete it or make your own code from scratch.
# 
# To compute the dencity estimation you can using the univariate version of kernel or the multivariate version of the kernel (use the one that you prefer). 

# In[ ]:


from kde import *


# In[ ]:


normalKernel1D(testSet[1])


# In[ ]:


normalKernelMultiD(testSet[1], c.shape[1])


# In[ ]:


# multivariate version of Kernel
normalKernelMultiD(testSet[1]-c[1],c.shape[1])
multiKernel = deltaMultiD(testSet[1],c[1],0.2,c.shape[1])


# In[ ]:


# univariate version of Kernel
normalKernel1D(testSet[1]-c[1])
uniKernel = deltaProd(testSet[1],c[1],0.2,c.shape[1])


# In[ ]:


# Check that the univariate version of kernel and the multivariate version produce the same results

difference = np.linalg.norm(uniKernel - multiKernel)
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The two methods give the same result')
else:
    print('Uh-oh! The two methods are different')


# In[ ]:


# Define parameters

# smoothing parameters
h = np.array([0.3,0.4,0.5])

# dimensionality of the test set
d = testSet.shape[1] 
print("dimensionality of the test set:"+str(d))


# ## Ex1. Plot p(x|ci) for every ci using h : 0.3, 0.4, 0.5 and comment your results. For a specific center (training point) comment how h influences the result.

# In[ ]:


#plot conditional density function
for j in range(len(h)):
    for i in range(len(c)):
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        ax.set_title("Kernel Function at point :  c = "+str(c[i])+ ". h : "+str(h[j]))
        ax.plot_surface(X,Y,densityEstimation().reshape(30,30))


# ## Ex1. Plot p(x)  using h : 0.3, 0.4, 0.5 and comment your results. Discuss the effect ofthe size of h.

# In[ ]:


#plot density function 
for i in range(len(h)):
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    ax.set_title("Density Function p(x), h:"+str(h[i]))
    ax.plot_surface(X,Y,densityEstimation().reshape(30,30))


# ## Ex2. Iris dataset: Using the functions that you created above (exercise 1) work with the iris dataset.
# * Plot the class conditional density of each attribute
# * For a given pair of attributes draw the two dimensional density for each class
# * Experiment with at least three different values of the h parameter and comment on your findings (in details).

# In[ ]:


# Load the Iris data.

from data_utils import load_IRIS



data_X, data_y = load_IRIS(test=False)

# As a sanity check, we print out the size of the data.
print('data shape: ', data_X.shape)
print('labels shape: ', data_y.shape)


# In[ ]:


# useful functions than you maybe wont to use (it's uo to you)
unique_y = np.unique (data_y)
points_by_class = [[x for x, t in zip (data_X, data_y) if t == c] for c in unique_y]
points_by_class_array = np.asarray(points_by_class)
points_by_class_array.shape


# ## Ex3. Iris dataset: Naive Bayes
#  Implement the Naive Bayes on iris dataset but now instead of assuming normal distribution estimate the probability distribution from the data using kernel density estimation with h : 0.3, 0.4, 0.5.
# * Discuss the effect of the h parameter in the accuracy of the algorithm
# * Compare with the results that you had in TP1 Naive Beyes
