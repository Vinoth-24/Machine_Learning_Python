#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest


# In[2]:


# # Load Dataset


# In[3]:


Customer = pd.read_csv('A:\Data Science by Excelr\Data science assignments\Hypothesis Testing\Costomer+OrderForm.csv')
Customer.head(10)


# In[4]:


Customer.shape


# In[5]:


Customer.dtypes


# In[6]:


Customer.info()


# In[7]:


Customer.describe()


# In[8]:


Phillippines_value=Customer['Phillippines'].value_counts()
print(Phillippines_value)


# In[9]:


Indonesia_value=Customer['Indonesia'].value_counts()
print(Indonesia_value)


# In[10]:


Malta_value=Customer['Malta'].value_counts()
print(Malta_value)


# In[11]:


India_value=Customer['India'].value_counts()
print(India_value)


# In[12]:


# # Hypothesis Testing


# In[13]:


chiStats = sp.stats.chi2_contingency([[271,267,269,280],[29,33,31,20]])


# In[14]:


print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))


# In[15]:


print('Interpret by p-Value')


# In[16]:


if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[17]:


# # Critical Value


# In[18]:


#critical value = 0.1
alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])
observed_chi_val = chiStats[0]


# In[19]:


print('Interpret by critical value')


# In[20]:


if observed_chi_val <= critical_value:
       print ('Null hypothesis cannot be rejected (variables are not related)')
else:
       print ('Null hypothesis cannot be excepted (variables are not independent)')


# In[21]:


#Inference is that proportion of defective % across the center is same.


# In[ ]:




