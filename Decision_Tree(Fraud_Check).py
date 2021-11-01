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


# In[4]:


Fraud_check= pd.read_csv("A:\Data Science by Excelr\Data science assignments\Decision Trees\Fraud_check.csv")
Fraud_check.head(20)


# In[5]:


Fraud_check.shape


# In[6]:


Fraud_check.dtypes


# In[7]:


Fraud_check.info()


# In[8]:


Fraud_check.describe()


# In[9]:


Fraud_check.corr()


# In[10]:


Fraud_check.columns


# In[11]:


Fraud_check.isnull().sum()


# In[13]:


# Visualization through sns
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(Fraud_check)


# In[14]:


sns.barplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[15]:


sns.boxplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[16]:


sns.lmplot(x='Taxable.Income',y='City.Population', data=Fraud_check)


# In[17]:


sns.jointplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[18]:


sns.stripplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[19]:


sns.distplot(Fraud_check['Taxable.Income'])


# In[20]:


sns.distplot(Fraud_check['City.Population'])


# In[21]:


# Preprocessing
Fraud_check.loc[Fraud_check["Taxable.Income"] <= 30000,"Taxable_Income"]="Good"
Fraud_check.loc[Fraud_check["Taxable.Income"] > 30001,"Taxable_Income"]="Risky"
#Fraud_check.loc[Fraud_check["Taxable.Income"]!="Good","Taxable_Income"]="Risky"


# In[22]:


Fraud_check


# In[23]:


# Label Encoding
label_encoder = preprocessing.LabelEncoder()
Fraud_check["Undergrad"] = label_encoder.fit_transform(Fraud_check["Undergrad"])
Fraud_check["Marital.Status"] = label_encoder.fit_transform(Fraud_check["Marital.Status"])
Fraud_check["Urban"] = label_encoder.fit_transform(Fraud_check["Urban"])
Fraud_check["Taxable_Income"] = label_encoder.fit_transform(Fraud_check["Taxable_Income"])


# In[24]:


Fraud_check.drop(['City.Population'],axis=1,inplace=True)
Fraud_check.drop(['Taxable.Income'],axis=1,inplace=True)


# In[25]:


Fraud_check["Taxable_Income"].unique()


# In[26]:


Fraud_check


# In[27]:


x = Fraud_check.iloc[:,0:4]
x


# In[28]:


y = Fraud_check["Taxable_Income"]
y


# In[29]:


len(y)


# In[30]:


colnames=list(Fraud_check.columns)
colnames


# In[31]:


# Split into train and Test Data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)


# In[32]:


model=DecisionTreeClassifier(criterion="gini")
model.fit(x_train,y_train)


# In[33]:


# Build Tree Model
tree.plot_tree(model)


# In[34]:


fn=[ 'Undergrad',
 'Marital.Status',
 'Taxable.Income',
 'Work.Experience',
   'Urban']
cn=['Good','Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,7), dpi=300)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[35]:


preds=model.predict(x_test)
pd.Series(preds).value_counts()


# In[36]:


pd.Series(y_test).value_counts()


# In[37]:


pd.crosstab(y_test,preds)


# In[38]:


np.mean(preds==y_test)


# In[39]:


array=Fraud_check.values
array


# In[40]:


X=array[:,0:4]
X


# In[41]:



Y=array[:,4]
Y


# In[42]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)


# In[43]:


from sklearn.tree import DecisionTreeRegressor
model1=DecisionTreeRegressor()


# In[44]:


model1.fit(X_train, Y_train)


# In[45]:


model1.score(X_test, Y_test)


# In[46]:


#The Regressor method is not best fit for Decision Tree


# In[ ]:




