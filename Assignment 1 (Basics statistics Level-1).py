#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Assignment-1-Q7 (Basic Statistics Level-1)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data1=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Basic Statistics_Level 1\Q7.csv")


# In[6]:


data1


# In[7]:


data1.mean() #mean


# In[8]:


data1.median()#median


# In[9]:


data1.Points.mode() #mode points


# In[10]:


data1.Score.mode() #mode sccore


# In[11]:


data1.Weigh.mode() #mode Weigh


# In[12]:


data1.var() #variance


# In[13]:


data1.std() #Standard deviation


# In[14]:


data1.describe() 


# In[22]:


Points_Range=data1.Points.max()-data1.Points.min() #range of points
Points_Range


# In[23]:


Score_Range=data1.Score.max()-data1.Score.min() #range of Score
Score_Range


# In[24]:


Weigh_Range=data1.Weigh.max()-data1.Weigh.min() #range of Weigh
Weigh_Range 


# In[25]:


#Box plot for inference


# In[18]:



f,ax=plt.subplots(figsize=(15,5))
plt.subplot(1,3,1)
plt.boxplot(data1.Points)
plt.title('Points')
plt.show()


# In[19]:


plt.subplot(1,3,2)
plt.boxplot(data1.Score)
plt.title('Score')
plt.show()


# In[20]:


plt.subplot(1,3,3)
plt.boxplot(data1.Weigh)
plt.title('Weigh')
plt.show()


# In[21]:


#Inferences: All data is concentrated around the median.
#For points Data set:
#No outliers can be seen, Distribution is right skewed
#for score Data set:
#3 Outliars can be seen: 5.250, 5.424, 5.345
#The distribution is left skewed
#For Weigh Data set:
#1 outlier is seen: 22.90
#the distribution is left skewed


# In[ ]:





# In[26]:


## Assignment-1-Q9_a (Basic Statistics Level-1)


# In[27]:


data2=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Basic Statistics_Level 1\Q9_a.csv")
data2


# In[31]:


data2.skew()#skewness


# In[32]:


data2.kurt()#kurtosis


# In[30]:


#Inferences:
# Skewness 1. Speed distribution is left skewed (negative skewness) 
#2. Distance distributin is right skewed (positive skewness)


# In[33]:


#Kurtosis: 1. Speed distribution is platykurtic (negative kurtosis)
#2. Distance distributin is leptokurtic (positive kurtosis)


# In[ ]:





# In[34]:


## Assignment-1-Q9_b (Basic Statistics Level-1)


# In[35]:


data3=pd.read_csv("A:\Data Science by Excelr\Data science assignments\Basic Statistics_Level 1\Q9_b.csv")
data3


# In[38]:


data3=data3.iloc[:,1:]


# In[40]:


data3


# In[47]:


data3.skew() #skewness


# In[48]:


data3.kurt() #kurtosis


# In[50]:


#Inferences
# Skewness: 1. WT distribution is left skewed (negative skewness) 
#2. SP distributin is right skewed (positive skewness)

#Kurtosis: SP & WT distribution both are leptokurtic (positive kurtosis) 


# In[ ]:





# In[51]:


## Assignment-1-Q11 (Basic Statistics Level-1)


# In[52]:


from scipy import stats
from scipy.stats import norm


# In[53]:


# Avg. weight of Adult in Mexico with 94% CI
stats.norm.interval(0.94,200,30/(2000**0.5))


# In[54]:


# Avg. weight of Adult in Mexico with 96% CI
stats.norm.interval(0.96,200,30/(2000**0.5))


# In[55]:


# Avg. weight of Adult in Mexico with 98% CI
stats.norm.interval(0.98,200,30/(2000**0.5))


# In[ ]:





# In[56]:


## Assignment-1-Q12 (Basic Statistics Level-1)


# In[57]:


x=pd.Series([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])
x


# In[58]:


sns.boxplot(x) #boxplot


# In[59]:


x.mean() #mean


# In[60]:


x.median() #median


# In[61]:


x.mode() #mode


# In[62]:


x.var() #variance


# In[63]:


x.std() #std


# In[64]:


plt.boxplot(x)


# In[65]:


#Inferences: Two outliers are found from the dataset at 49 and 56.


# In[ ]:





# In[66]:


## Assignment-1-Q20 (Basic Statistics Level-1)


# In[67]:


cars=pd.read_csv('A:\Data Science by Excelr\Data science assignments\Basic Statistics_Level 1\Cars.csv')
cars


# In[68]:


cars.describe()


# In[ ]:





# In[73]:


cars.MPG.mean()


# In[70]:


cars.MPG.std()


# In[74]:


# P(MPG>38)
stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())


# In[75]:


# P(MPG<40)
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())


# In[76]:


# P (20<MPG<50)
stats.norm.cdf(50,cars.MPG.mean(),cars.MPG.std())-stats.norm.cdf(20,cars.MPG.mean(),cars.MPG.std())


# In[ ]:





# In[77]:


## Assignment-1-Q21_a (Basic Statistics Level-1)


# In[78]:


sns.distplot(cars.MPG, label='Cars-MPG')
plt.xlabel('MPG')
plt.ylabel('Density')
plt.legend();


# In[79]:


cars.MPG.mean()


# In[80]:


cars.MPG.median()


# In[82]:


#Inference: The MPG of cars follow normal distribution as the mean and median values are approximately same.


# In[ ]:





# In[83]:


## Assignment-1-Q21_b (Basic Statistics Level-1)


# In[84]:


wcat=pd.read_csv('A:\Data Science by Excelr\Data science assignments\Basic Statistics_Level 1\wc-at.csv')
wcat


# In[91]:


# plotting distribution for Waist Tissue (WT)
sns.distplot(wcat.Waist) 
plt.ylabel('density')
plt.show()


# In[97]:


# plotting distribution for Adipose Tissue (AT)
sns.distplot(wcat.AT)
plt.ylabel('density')
plt.show()


# In[93]:


wcat.Waist.mean()  


# In[94]:


wcat.Waist.median()


# In[95]:


wcat.AT.mean() 


# In[96]:


wcat.AT.median()


# In[98]:


#Inference: Both AT and Waist dont follow normal distribution as their mean and median values are different


# In[ ]:





# In[99]:


## Assignment-1-Q22 (Basic Statistics Level-1)


# In[100]:


from scipy import stats
from scipy.stats import norm


# In[104]:


# Z-score of 90% confidence interval 
stats.norm.ppf(0.90)


# In[102]:


# Z-score of 94% confidence interval
stats.norm.ppf(0.94)


# In[103]:


# Z-score of 60% confidence interval
stats.norm.ppf(0.60)


# In[ ]:





# In[105]:


## Assignment-1-Q23 (Basic Statistics Level-1)


# In[106]:


n=25
df=n-1
df


# In[107]:


#t scores of 95% confidence interval for sample size of 25
stats.t.ppf(0.95,24)


# In[108]:


#t scores of 96% confidence interval for sample size of 25
stats.t.ppf(0.96,24)


# In[109]:


#t scores of 99% confidence interval for sample size of 25
stats.t.ppf(0.99,24)


# In[ ]:





# In[125]:


## Assignment-1-Q24 (Basic Statistics Level-1)


# In[111]:


#Pop mean =270 days, sample mean = 260 days, Sample SD = 90 days, Sample n = 18 bulbs


# In[112]:


n = 18
df = n-1 
df


# In[113]:


# Assume Null Hypothesis is: Ho = Avg life of Bulb >= 260 days
# Alternate Hypothesis is: Ha = Avg life of Bulb < 260 days


# In[114]:


# finding t-scores at x=260; t=(s_mean-P_mean)/(s_SD/sqrt(n))
t=(260-270)/(90/18**0.5)
t


# In[123]:


# p_value=1-stats.t.cdf(abs(t_scores),df=n-1)... Using cummulative distributive function
p_value=1-stats.t.cdf(abs(-0.4714),df=17)
p_value


# In[124]:


#  OR p_value=stats.t.sf(abs(t_score),df=n-1)... Using survival function
p_value=stats.t.sf(abs(-0.4714),df=17)
p_value


# In[118]:


#Probability that 18 randomly selected bulbs would have an average life of no more than 260 days is 32.17%
#Assuming significance value α = 0.05 (Standard Value)(If p_value < α ; Reject Ho and accept Ha or vice-versa)
#Thus, as p-value > α ; Accept Ho i.e. The CEO claims are false and the avg life of bulb > 260 days


# In[ ]:





# In[ ]:




