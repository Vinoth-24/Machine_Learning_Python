#!/usr/bin/env python
# coding: utf-8

# In[6]:


#importing libraries


# In[3]:


import pandas as pd


# In[4]:


import seaborn as sns


# In[5]:


import numpy as np


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


#reading csv files


# In[8]:


#reading csv files


# In[9]:


D_time=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Simple Linear Regression\delivery_time.csv")


# In[10]:


#checking null values


# In[11]:


null = D_time.isnull()


# In[12]:


null


# In[13]:


#Renaming dataset 


# In[14]:


D_time.rename(columns={'Delivery Time' :'D' , 'Sorting Time':'S'}, inplace="True")


# In[15]:


D_time


# In[16]:


#plot to understand data


# In[17]:


sns.pairplot(D_time)


# In[28]:


plt.hist(D_time.D)


# In[29]:


plt.hist(D_time.S)


# In[30]:


plt.boxplot(D_time.D)


# In[31]:


plt.boxplot(D_time.S)


# In[32]:


#checking correlation b/w D and S


# In[26]:


corrMatrix=D_time.corr()


# In[27]:


D_time.D.corr(D_time.S)


# In[28]:


sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[ ]:





# In[34]:


import statsmodels.formula.api as smf


# In[35]:


#model building and fitting


# In[36]:


model=smf.ols("D~S", data=D_time).fit()


# In[32]:


sns.regplot(x="S", y="D", data=D_time);


# In[39]:


model.summary()


# In[41]:


model.conf_int(0.05) #data spread in the range


# In[43]:


pred = model.predict(D_time.iloc[:,1]) #predicting values for D with S


# In[58]:


plt.scatter(D_time.S,D_time.D,c='b');plt.plot(D_time.S,pred,c='r');plt.xlabel("Sorting time");plt.ylabel("Delivery Time") #regression line visualization


# In[48]:


pred.corr(D_time.D)


# In[49]:


#using log in model building for better correlation


# In[51]:


model2 = smf.ols('D~np.log(S)',data=D_time).fit()


# In[53]:


model2.summary()


# In[55]:


pred2 = model2.predict(D_time.S)


# In[57]:


pred2.corr(D_time.D) #increase in correlation noticed


# In[59]:


plt.scatter(D_time.S, D_time.D, c='b');plt.plot(D_time.S,pred2,c='r');plt.xlabel("Sorting time");plt.ylabel("Delivery Time")


# In[61]:


model3=smf.ols('np.log(D)~S',data=D_time).fit()


# In[62]:


model3.summary()


# In[65]:


pred_log=model3.predict(D_time.S)


# In[66]:


pred_log


# In[67]:


pred3=np.exp(pred_log)


# In[70]:


pred3.corr(D_time.D)


# In[69]:


plt.scatter(D_time.S,D_time.D,c='b');plt.plot(D_time.S,pred3,c='r');plt.xlabel("Sorting time");plt.ylabel("Delivery Time")


# In[75]:


#model2 has the highest correlation. so we select it and find residual errors and plot them to check whether they are normally distributed


# In[81]:


errors= pred2-D_time.D


# In[82]:


normalized_resid = model2.resid_pearson 


# In[83]:


normalized_resid


# In[84]:


plt.plot(normalized_resid,'o');plt.axhline(y=0,color='green');plt.xlabel("Data points");plt.ylabel("Normalized Residual")


# In[80]:


#predicted vs actual values


# In[87]:


plt.scatter(x=pred2,y=D_time.D);plt.xlabel("Predicted");plt.ylabel("Actual")


# In[88]:


model_influence=model2.get_influence()


# In[89]:


model_influence


# In[108]:


c,_ = model_influence.cooks_distance


# In[109]:


print (c)


# In[111]:


plt.scatter(D_time.S, c)


# In[135]:


#we are dropping the 5th row because it could be due to some issues during sorting time


# In[122]:


np.argmax(c),np.max(c)


# In[127]:


new=D_time.drop(columns='ID')


# In[130]:


final=new.reset_index()


# In[131]:


final=D_time.drop(index=4)


# In[132]:


final_model=smf.ols('D~np.log(S)',data=final).fit()


# In[134]:


final_model.summary()


# In[136]:


final_predict=final_model.predict(final.S)


# In[137]:


final_predict.corr(final.D)    #the correlation and sqr R has improved 


# The correlation has improved and sqr R has also improved in the final_model. We have also checked for the errors to be normally distributed. Now we can deploy this final_model.

# In[138]:


plt.scatter(final.S,final.D,c='b');plt.plot(final.S,final_predict,c='r');plt.xlabel("Sorting time");plt.ylabel("Delivery Time")


# In[ ]:




