#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns


# In[ ]:


# load Data 
data=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Simple Linear Regression\Salary_Data.csv")
data


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# Graphical Representation of Data
data.plot()


# In[ ]:


data.corr()


# In[ ]:


data.Salary


# In[ ]:


data.YearsExperience


# In[ ]:


sns.distplot(data['Salary'])


# In[ ]:


sns.distplot(data['YearsExperience'])


# In[ ]:


sns.pairplot(data)


# In[ ]:


sns.scatterplot(x=data.YearsExperience, y=np.log(data.Salary), data=data)


# In[ ]:


# Calculate R^2 values
import statsmodels.formula.api as smf
import pandas.util.testing as tm
model = smf.ols("Salary~YearsExperience",data = data).fit()


# In[ ]:


sns.regplot(x="Salary", y="YearsExperience", data=data)


# In[ ]:


#Coefficients
model.params


# In[ ]:


model =smf.ols('Salary~YearsExperience', data=data).fit()
model


# In[ ]:


model.summary()


# In[ ]:


#Predict for 15 and 20 Year's of Experiance 
newdata=pd.Series([15,20])


# In[ ]:


data_pred=pd.DataFrame(newdata,columns=['YearsExperience'])
data_pred


# In[ ]:


model.predict(data_pred).round(2)


# In[ ]:




