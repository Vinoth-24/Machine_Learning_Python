#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[54]:


# Load Dataset 
Air=pd.read_excel("A:\Data Science by Excelr\Data science assignments\Forecasting\Airlines+Data.xlsx")
Air


# In[3]:


Air['Month'].dtypes


# In[4]:


Air.tail()


# In[5]:


Air.Passengers.plot()


# In[6]:


Air.columns = ['ds','y']
Air.head()


# In[7]:


Air['ds'] = pd.to_datetime(Air['ds'])


# In[12]:


from fbprophet import Prophet


# In[13]:


dir(Prophet)


# In[14]:


# Initialize the Model
model=Prophet()


# In[29]:


help(model)


# In[16]:


Air.columns


# In[17]:


model.fit(Air)


# In[18]:


model.component_modes


# In[20]:


Air.tail()


# In[39]:


### Create future dates of 365 days
future_dates=model.make_future_dataframe(freq = "MS",periods=12)


# In[40]:


future_dates.tail()


# In[41]:


prediction=model.predict(future_dates)


# In[42]:


prediction.head()


# In[43]:


# plot the predicted projection
model.plot(prediction)


# In[44]:


# Visualize Each Components[Trends,yearly]
model.plot_components(prediction)


# In[45]:


from fbprophet.diagnostics import cross_validation
Air_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
Air_cv.head()


# In[53]:


Air_cv


# In[46]:


from fbprophet.diagnostics import performance_metrics
Air_p = performance_metrics(Air_cv)
Air_p.head()


# In[48]:


Air_p


# In[47]:


from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(Air_cv, metric='rmse')

