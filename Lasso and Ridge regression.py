#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Reading the file

# In[1]:


import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# In[2]:


data=pd.read_csv("A:\\Data Science by Excelr\\PROJECT\\bankruptcy_prevention.csv",sep=";")


# In[3]:


data.head()


# In[4]:


data=data.rename(columns = {data.columns[6]: 'class_value'})
data=data.rename(columns = {data.columns[5]: 'operating_risk'})
data=data.rename(columns = {data.columns[4]: 'competitiveness'})
data=data.rename(columns = {data.columns[3]: 'credibility'})
data=data.rename(columns = {data.columns[2]: 'financial_flexibility'})
data=data.rename(columns = {data.columns[1]: 'management_risk'})


# In[5]:


label_encoder = preprocessing.LabelEncoder()
data["class_value"] = label_encoder.fit_transform(data["class_value"])


# In[6]:


data.head()


# In[7]:


features=data.columns[0:6]
features


# In[8]:


# Model Building Starts


# In[9]:


X=data[features]


# In[10]:


Y=data['class_value']


# In[11]:


#pip install mljar-supervised --user


# In[12]:


#from supervised.automl import AutoML


# In[13]:


# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2)


# In[14]:


#automl.fit(X_train, y_train)


# In[ ]:


#automl.report()


# In[15]:


# Regression


# In[20]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X,Y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[21]:


# Ridge


# In[22]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,Y)


# In[23]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[24]:


#Lasso


# In[27]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-55, 1e-50, 1e-45, 1e-40, 1e-35, 1e-30, 1e-25, 1e-20, 1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,Y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[28]:


prediction_lasso=lasso_regressor.predict(X_test)
prediction_ridge=ridge_regressor.predict(X_test)


# In[31]:


import seaborn as sns

sns.distplot(y_test-prediction_lasso)


# In[32]:


sns.distplot(y_test-prediction_ridge)


# In[33]:


# NO DIFFERENCE IN ERROR NOTICED USING REGULARIZATION


# In[ ]:




