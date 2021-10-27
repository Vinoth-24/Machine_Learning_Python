#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


Startups=pd.read_csv("A:\\Data Science by Excelr\\Data science assignments\\Multi Linear Regression\\50_Startups.csv")


# In[ ]:


#reading csv file


# In[ ]:


Startups.head(10)


# In[ ]:


null = Startups.isnull() #checking for null values


# In[ ]:


null


# In[ ]:


Startups.corr()


# In[ ]:


ax=Startups.groupby(['State'])['Profit'].mean().plot.bar(figsize=(10,5), fontsize=14)


# In[ ]:


Start=pd.get_dummies(Startups)  #using dummies for Categorical value


# In[ ]:


Start.head()


# In[ ]:


import seaborn as sns # Scatter plot between the variables along with histograms


# In[ ]:


sns.pairplot(Startups)


# In[ ]:


Start.columns #renaming dataset


# In[ ]:


Start.rename(columns={'R&D Spend' :'Research' , 'Administration':'Admin', 'Marketing Spend':'Market','State_New York':'State_NewYork' }, inplace="True")


# In[ ]:


Start.drop(columns='State_NewYork',inplace=True)  #dropping the the 3rd dummy variable


# In[ ]:


#Using ordinary Least squares technique
import statsmodels.formula.api as smf


# In[ ]:


M1= smf.ols('Profit~Research+Admin+Market+State_California+State_Florida', data= Start).fit()


# In[ ]:


M1.params


# In[ ]:


M1.summary() #Adjusted R-sqr= 0.95


# In[ ]:


Ml_A=smf.ols('Profit~Admin',data = Start).fit()  


# In[ ]:


Ml_A.summary()  #Administration has very low R-sqr value= 0.04 and P value>0.05( insignificant)


# In[ ]:


M1_M=smf.ols('Profit~Market', data=Start).fit()


# In[ ]:


M1_M.summary() # Marketting spend has good R-squared=0.559 and p-value<0.05 (significant)


# In[ ]:


M1_Cali=smf.ols('Profit~State_California', data=Start).fit()


# In[ ]:


M1_Cali.summary()  #low R-sqr value and pvalue>0.05 (insignificant)


# In[ ]:


M1_Flo=smf.ols('Profit~State_Florida', data=Start).fit()


# In[ ]:


M1_Flo.summary()  #low R-sqr value and pvalue>0.05 (insignificant)


# In[ ]:


Ml_AandM=smf.ols('Profit~Admin+Market',data = Start).fit() 


# In[ ]:


Ml_AandM.summary() #now pvalue seems better but highly dominant by Market on R-sqr value


# In[ ]:


import statsmodels.api as sm #importing influence plots


# In[ ]:


sm.graphics.influence_plot(M1) #remove the rows with influencers


# In[ ]:


Start_new=Start.drop(Start.index[[45,48,49]],axis=0)


# In[ ]:


Start


# In[ ]:


Start_new


# In[ ]:


X=Start.drop(columns="Profit")


# In[ ]:


Y=Start.drop(columns=["Research","Admin","Market","State_California", "State_Florida"])


# In[ ]:


M1_new= smf.ols('Profit~Research+Admin+Market+State_California+State_Florida', data= Start_new).fit()


# In[ ]:


M1_new.summary() #change in R-squared values. 


# In[ ]:


# calculating VIF's values of independent variables


# In[ ]:


rsq_R = smf.ols('Research~Admin+Market+State_California+State_Florida',data=Start_new).fit().rsquared  


# In[ ]:


vif_R = 1/(1-rsq_R)


# In[ ]:


rsq_A = smf.ols('Admin~Research+Market+State_California+State_Florida',data=Start_new).fit().rsquared


# In[ ]:


vif_A = 1/(1-rsq_A)


# In[ ]:


rsq_M = smf.ols('Market~Research+Admin+State_California+State_Florida',data=Start_new).fit().rsquared


# In[ ]:


vif_M = 1/(1-rsq_M)


# In[ ]:


rsq_Cal = smf.ols('State_California~Market+Research+Admin+State_Florida',data=Start_new).fit().rsquared


# In[ ]:


vif_Cal = 1/(1-rsq_Cal)


# In[ ]:


rsq_Flo = smf.ols('State_Florida~State_California+Market+Research+Admin+State_California',data=Start_new).fit().rsquared


# In[ ]:


vif_Flo = 1/(1-rsq_Flo)


# In[ ]:


d1 = {'Variables':['Research','Admin','Market','State_California','State_Florida'],'VIF':[vif_R,vif_A,vif_M,vif_Cal,vif_Flo]}


# In[ ]:


pd.DataFrame(d1) #correlation of independent variable are lower


# In[ ]:


#lets try removing Admin because it has low R-sqr value


# In[ ]:


M2=smf.ols('Profit~Research+Market+State_California+State_Florida',data=Start_new).fit()


# In[ ]:


M2.summary() #removing the influencers and the Admin column has resulted in a better model with R-sqr=0.963


# In[ ]:


sm.graphics.plot_partregress_grid(M2)


# In[ ]:


#Now we can start predicting data 


# In[ ]:


Profit_pred=M2.predict(Start_new)


# In[ ]:


Start_new.Profit


# In[ ]:


Profit_pred


# In[ ]:


Profit_pred.corr(Start_new.Profit) #very high correlation b/2 predicted Profit vs Actual Profit


# In[ ]:


#Fitting Observed vs Fitted values of Profit


# In[ ]:


plt.scatter(Start_new.Profit,Profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")


# In[ ]:


#Residuals vs Fitted Values #we can see errors are normally distributed


# In[ ]:


plt.scatter(Profit_pred,M2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# In[ ]:


plt.hist(M2.resid_pearson)#Normally distributed


# In[ ]:


from sklearn.model_selection import train_test_split  #now using train_test_split to create model and test on unknown data


# In[ ]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size = 0.2)


# In[ ]:


X_train


# In[ ]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()


# In[ ]:


clf.fit(X_train,Y_train)


# In[ ]:


clf.predict(X_test)


# In[ ]:


clf.score(X_test,Y_test)

