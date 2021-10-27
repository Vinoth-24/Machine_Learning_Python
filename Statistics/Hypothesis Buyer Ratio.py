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


# In[13]:


#Load data
BuyerRatio =pd.read_csv('A:\Data Science by Excelr\Data science assignments\Hypothesis Testing\BuyerRatio.csv')
BuyerRatio.head(10)


# In[3]:


BuyerRatio.shape


# In[4]:


BuyerRatio.dtypes


# In[5]:


BuyerRatio.info()


# In[6]:


BuyerRatio.describe()


# In[7]:


East=BuyerRatio['East'].mean()
print('East Mean = ',East)


# In[8]:


West=BuyerRatio['West'].mean()
print('West Mean = ',West)


# In[9]:


North=BuyerRatio['North'].mean()
print('North Mean = ',North)


# In[10]:


South=BuyerRatio['South'].mean()
print('South Mean = ',South)


# In[11]:


#The Null and Alternative Hypothesis

#There are no significant differences between the groups' mean values. H0:μ1=μ2=μ3=μ4=μ5

#There is a significant difference between the groups' mean values. Ha:μ1≠μ2≠μ3≠μ4


# In[12]:


## Visualization


# In[14]:


sns.distplot(BuyerRatio['East'])
sns.distplot(BuyerRatio['West'])
sns.distplot(BuyerRatio['North'])
sns.distplot(BuyerRatio['South'])
plt.legend(['East','West','North','South'])


# In[15]:


sns.boxplot(data=[BuyerRatio['East'],BuyerRatio['West'],BuyerRatio['North'],BuyerRatio['South']],notch=True)
plt.legend(['East','West','North','South'])


# In[16]:


## Hypothesis Testing


# In[17]:


alpha=0.05
Male = [50,142,131,70]
Female=[435,1523,1356,750]
Sales=[Male,Female]
print(Sales)


# In[18]:


chiStats = sp.stats.chi2_contingency(Sales)
print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))
print('Interpret by p-Value')


# In[19]:


if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[20]:


alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])
critical_value 


# In[22]:


## Degree of Freedom


# In[23]:


observed_chi_val = chiStats[0]
print('Interpret by critical value')


# In[24]:


if observed_chi_val <= critical_value:
    print ('Null hypothesis cannot be rejected (variables are not related)')
else:
    print ('Null hypothesis cannot be excepted (variables are not independent)')


# In[25]:


#Inference : proportion of male and female across regions is same


# In[ ]:




