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


# In[4]:


#load data
Data=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Hypothesis Testing\Cutlets.csv")


# In[6]:


Data.head(10)


# In[7]:


Data.shape


# In[9]:


Data.dtypes


# In[10]:


#Data.isnull()


# In[11]:


Data.info()


# In[15]:


Data.describe()


# In[16]:


UnitA=Data['Unit A'].mean()


# In[17]:


UnitB=Data['Unit B']. mean()


# In[30]:


print("Unit A mean: ", UnitA, "Unit B mean:", UnitB )


# In[31]:


UnitA > UnitB


# In[32]:


#Visualization


# In[33]:


sns.distplot(Data['Unit A'])


# In[34]:


sns.distplot(Data['Unit B'])


# In[36]:


sns.distplot(Data['Unit A'])
sns.distplot(Data['Unit B'])
plt.legend(['Unit A','Unit B'])


# In[40]:


sns.boxplot(data=[Data['Unit A'],Data['Unit B']])
plt.legend(['Unit A','Unit B'])


# In[41]:


## Hypothesis Testing


# In[42]:


alpha=0.05
Unit_A=pd.DataFrame(Data['Unit A'])
Unit_A


# In[45]:


Unit_B=pd.DataFrame(Data['Unit B'])
Unit_B


# In[47]:


tStat,pValue =sp.stats.ttest_ind(Unit_A,Unit_B)


# In[54]:


print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat))


# In[51]:


if pValue <0.05:
  print('we reject null hypothesis')
else:
  print('we fail to reject null hypothesis')


# In[50]:


#Inference is that there is no significant difference in the diameters of Unit A and Unit B


# In[ ]:




