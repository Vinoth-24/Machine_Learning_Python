#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load Dataset
fraud=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Random Forests\Fraud_check.csv")
fraud.head(10)


# In[3]:


fraud.dtypes


# In[4]:


fraud.info()


# In[5]:


fraud.columns


# In[6]:


fraud.shape


# In[7]:


fraud.isnull().sum()


# In[8]:


fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
fraud["TaxInc"]


# In[10]:


fraudcheck = fraud.drop(columns=["Taxable.Income"])
fraudcheck 


# In[11]:


FC = pd.get_dummies(fraudcheck .drop(columns = ["TaxInc"]))


# In[12]:


FC


# In[13]:


Fraud_final = pd.concat([FC,fraudcheck ["TaxInc"]], axis = 1)


# In[14]:


Fraud_final


# In[15]:


colnames = list(Fraud_final.columns)
colnames


# In[16]:


predictors = colnames[:9]
predictors


# In[17]:


target = colnames[9]
target


# In[18]:


X = Fraud_final[predictors]
X.shape


# In[45]:


Y=Fraud_final[target]


# In[46]:


Y


# In[19]:


# Visualization
sns.pairplot(fraud)


# In[20]:


sns.barplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[21]:


sns.boxplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[22]:


sns.lmplot(x='Taxable.Income',y='City.Population', data=fraud)


# In[23]:


sns.jointplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[24]:


sns.stripplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[25]:


sns.distplot(fraud['Taxable.Income'])


# In[26]:


sns.distplot(fraud['City.Population'])


# In[34]:


# Building Random Forest Model
from sklearn.ensemble import RandomForestClassifier


# In[35]:


rf = RandomForestClassifier(n_jobs = 3, oob_score = True, n_estimators = 15, criterion = "entropy")


# In[36]:


np.shape(Fraud_final)


# In[37]:


Fraud_final.describe()


# In[38]:


Fraud_final.info()


# In[47]:


type([X])


# In[48]:


type([Y])


# In[49]:


Y1 = pd.DataFrame(Y)
Y1


# In[50]:


rf.fit(X,Y1)


# In[51]:


rf.estimators_ 


# In[52]:


rf.classes_


# In[53]:


rf.n_classes_ 


# In[54]:


rf.n_features_


# In[55]:


rf.n_outputs_


# In[56]:


rf.oob_score_ 


# In[57]:


rf.predict(X)


# In[58]:


Fraud_final['rf_pred'] = rf.predict(X)


# In[59]:


cols = ['rf_pred','TaxInc']


# In[60]:


Fraud_final[cols].head()


# In[68]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[69]:


confusion_matrix(Fraud_final['TaxInc'],Fraud_final['rf_pred'])


# In[70]:


pd.crosstab(Fraud_final['TaxInc'],Fraud_final['rf_pred'])


# In[72]:


score=accuracy_score(Fraud_final['rf_pred'],Fraud_final['TaxInc'])
score


# In[73]:


print("Accuracy",score*100)


# In[74]:


Fraud_final["rf_pred"]


# In[ ]:




