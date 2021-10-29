#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load Dataset
Coca=pd.read_excel("A:\Data Science by Excelr\Data science assignments\Forecasting\CocaCola_Sales_Rawdata.xlsx")
Coca.head(10)


# In[3]:


Coca.shape


# In[4]:


Coca.dtypes


# In[5]:


Coca.info()


# In[6]:


Coca.describe()


# In[7]:


# Visualization
sns.boxplot("Sales",data=Coca)


# In[8]:


sns.catplot("Quarter","Sales",data=Coca,kind="box")


# In[9]:


Coca.Sales.plot(label="org")
for i in range(2,10,2):
    Coca["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[10]:


Coca.Sales.plot()


# In[11]:


plt.figure(figsize=(24,5))
Coca.Sales.plot()


# In[12]:


# Preprocessing
quarter=['Q1','Q2','Q3','Q4']
n=Coca['Quarter'][0]
n[0:2]


# In[15]:


Coca['quarter']=0
for i in range(42):
    n=Coca['Quarter'][i]
    Coca['quarter'][i]=n[0:2]
    dummy=pd.DataFrame(pd.get_dummies(Coca['quarter']))
    coco=pd.concat((Coca,dummy),axis=1)
t= np.arange(1,43)
coco['t']=t
coco['t_square']=coco['t']*coco['t'] 
log_Sales=np.log(coco['Sales'])  #for stabilizing the time series
coco['log_Sales']=log_Sales


# In[16]:


coco


# In[17]:


plt.figure(figsize=(12,3))
sns.lineplot(x="quarter",y="Sales",data=Coca)


# In[18]:


# Splitting Data
Train = coco.head(34)


# In[19]:


Test = coco.iloc[34:38,:]


# In[20]:


predict_data = coco.tail(4)


# In[21]:


Coca2= coco.iloc[0:38,:]


# In[22]:


Train


# In[23]:


Test


# In[24]:


predict_data


# In[25]:


# Build Model & Calculate RMSE Values
# Using Linear Model
import statsmodels.formula.api as smf 
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[26]:


# Exponential
Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[27]:


# Quadratic 
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[28]:


# Additive Seasonality Quadratic
add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[29]:


# Multiplicative Seasonality
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[30]:


# Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# In[33]:


# Compare the results 
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[34]:


# Predict the New Model
predict_data


# In[35]:


# Build the model on entire data set
model_full = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Coca2).fit()


# In[36]:


pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new


# In[37]:


predict_data["forecasted_Sales"] = pd.DataFrame(pred_new)


# In[38]:


predict_data


# In[ ]:




