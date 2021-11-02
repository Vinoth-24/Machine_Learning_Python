#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Load Dataset
company=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Random Forests\company_data.csv")
company.head(10)


# In[7]:


company.shape


# In[8]:


company.dtypes


# In[9]:


company.info()


# In[14]:


# Converting from Categorical Data
company['High'] = company.Sales.map(lambda x: 1 if x>company['Sales'].mean() else 0)


# In[15]:


company['High']


# In[16]:


company['ShelveLoc']=company['ShelveLoc'].astype('category')


# In[17]:


company['ShelveLoc']


# In[18]:


company['Urban']=company['Urban'].astype('category')


# In[19]:


company['Urban']


# In[20]:


company['US']=company['US'].astype('category')


# In[21]:


company.dtypes


# In[22]:


company.head(10)


# In[23]:


# label encoding to convert categorical values into numeric.
company['ShelveLoc']=company['ShelveLoc'].cat.codes
company['ShelveLoc']


# In[24]:


company['Urban']=company['Urban'].cat.codes


# In[25]:


company['US']=company['US'].cat.codes


# In[26]:


company.head(10)


# In[27]:


# Visualization
sns.pairplot(company)


# In[28]:


sns.barplot(company['Sales'], company['Income'])


# In[29]:


sns.boxplot(company['Sales'], company['Income'])


# In[30]:


sns.lmplot(x='Income', y='Sales', data=company)


# In[31]:


sns.jointplot(company['Sales'], company['Income'])


# In[32]:


sns.stripplot(company['Sales'], company['Income'])


# In[33]:


sns.distplot(company['Sales'])


# In[34]:


sns.distplot(company['Income'])


# In[35]:


# setting feature and target variables
feature_cols=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']


# In[36]:


x = company.drop(['Sales', 'High'], axis = 1)


# In[37]:


x = company[feature_cols]


# In[39]:


y = company.High


# In[40]:


x


# In[41]:


y


# In[42]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split


# In[43]:


x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[44]:


x_train


# In[45]:


y_train


# In[46]:


x_valid


# In[47]:


y_valid


# In[48]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[49]:


sc = StandardScaler()


# In[50]:


x_train = sc.fit_transform(x_train)


# In[52]:


x_valid = sc.transform(x_valid)


# In[53]:


x_train


# In[54]:


x_valid


# In[55]:


# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[56]:


preds=classifier.predict(x_valid)


# In[57]:


preds


# In[59]:


from sklearn.metrics import mean_absolute_error


# In[60]:


mae=mean_absolute_error(preds,y_valid)


# In[61]:


mae


# In[63]:


classifier.score(x_valid, y_valid)


# In[64]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score


# In[67]:


cm = confusion_matrix(y_valid, preds)


# In[68]:


cm


# In[69]:


accuracy_score(y_valid, preds)


# In[70]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
classifier.fit(x_train, y_train)


# In[72]:


classifier.score(x_valid, y_valid)


# In[ ]:




