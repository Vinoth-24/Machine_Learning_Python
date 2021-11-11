#!/usr/bin/env python
# coding: utf-8

# In[66]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[67]:


# Load Dataset
fire= pd.read_csv(r"A:\Data Science by Excelr\Data science assignments\Support Vector Machines\forestfires.csv")
fire.head(10)


# In[68]:


# Preprocessing & Label Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
fire["month"] = label_encoder.fit_transform(fire["month"])
fire["day"] = label_encoder.fit_transform(fire["day"])
fire["size_category"] = label_encoder.fit_transform(fire["size_category"])


# In[69]:


fire


# In[70]:


# Visualization
for i in fire.describe().columns[:-2]:
    fire.plot.scatter(i,'area',grid=True)


# In[71]:


fire.groupby('day').area.mean().plot(kind='bar')


# In[72]:


fire.groupby('day').area.mean().plot(kind='box')


# In[73]:


fire.groupby('month').area.mean().plot(kind='box')


# In[74]:


fire.groupby('day').area.mean().plot(kind='line')


# In[75]:


fire.groupby('day').area.mean().plot(kind='hist')


# In[76]:


fire.groupby('day').area.mean().plot(kind='density')


# In[77]:


X=fire.iloc[:,:11]
X


# In[13]:


y=fire["size_category"]
y


# In[78]:


#checking with temp and wind on y with 3D graph with plotly
import plotly.express as px

fig = px.scatter_3d(fire, x='temp', y='wind', z='size_category',
              color='temp')
fig.show()


# In[79]:


# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[80]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[81]:


# Grid Search CV
clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[82]:


gsv.best_params_ , gsv.best_score_


# In[83]:


clf = SVC(C= 15, gamma = 0.5)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[84]:


clf1 = SVC(C= 15, gamma = 50)
clf1.fit(X , y)
y_pred = clf1.predict(X)
acc1 = accuracy_score(y, y_pred) * 100
print("Accuracy =", acc1)
confusion_matrix(y, y_pred)


# In[85]:


# Poly
clf2 = SVC()
param_grid = [{'kernel':['poly'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[86]:


gsv.best_params_ , gsv.best_score_


# In[87]:


clf2_= SVC(C= 15, gamma = 50)
clf2_.fit(X_train , y_train)
y_pred = clf2_.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[88]:


# Sigmoid
clf3 = SVC()
param_grid = [{'kernel':['sigmoid'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[89]:


gsv.best_params_ , gsv.best_score_


# In[90]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[ ]:




