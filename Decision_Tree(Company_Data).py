#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree 
from sklearn.metrics import classification_report 
from sklearn import preprocessing


# In[3]:


# Load Dataset
Comp_Data= pd.read_csv("A:\Data Science by Excelr\Data science assignments\Decision Trees\Company_Data.csv")
Comp_Data.head(20)


# In[4]:


Comp_Data.shape


# In[5]:


Comp_Data.info()


# In[6]:


Comp_Data.dtypes


# In[7]:


Comp_Data.describe()


# In[8]:


Comp_Data.corr()


# In[9]:


# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(Comp_Data)


# In[10]:


sns.barplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[11]:


sns.boxplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[12]:


sns.lmplot(x='Income', y='Sales', data=Comp_Data)


# In[13]:


sns.jointplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[14]:


sns.swarmplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[15]:


sns.distplot(Comp_Data['Sales'])


# In[16]:


sns.distplot(Comp_Data['Income'])


# In[17]:


# Preprocessing
Comp_Data.loc[Comp_Data["Sales"] <= 10.00,"Sales1"]="Not High"
Comp_Data.loc[Comp_Data["Sales"] >= 10.01,"Sales1"]="High"


# In[18]:


Comp_Data


# In[19]:


# Label Encoding
label_encoder = preprocessing.LabelEncoder()
Comp_Data["ShelveLoc"] = label_encoder.fit_transform(Comp_Data["ShelveLoc"])
Comp_Data["Urban"] = label_encoder.fit_transform(Comp_Data["Urban"])
Comp_Data["US"] = label_encoder.fit_transform(Comp_Data["US"])
Comp_Data["Sales1"] = label_encoder.fit_transform(Comp_Data["Sales1"])


# In[20]:


Comp_Data


# In[21]:


x=Comp_Data.iloc[:,1:11]
x


# In[22]:


y=Comp_Data["Sales1"]
y


# In[23]:


Comp_Data.Sales1.value_counts()


# In[24]:


colnames=list(Comp_Data.columns)
colnames


# In[25]:


# Split Data into Train and Test 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[26]:


model = DecisionTreeClassifier(criterion = 'entropy')


# In[27]:


model.fit(x_train,y_train)


# In[28]:


# Build Tree Model
tree.plot_tree(model);


# In[29]:


fn=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
cn=['Not High Sales', 'High Sales']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,7), dpi=600)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[30]:


preds=model.predict(x_test)
pd.Series(preds).value_counts()


# In[31]:


preds=model.predict(x_test)
pd.Series(preds).value_counts()
pd.Series(y_test).value_counts()


# In[32]:


pd.crosstab(y_test,preds)


# In[33]:


np.mean(preds==y_test)








