#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(r'A:\Ineuron\ml-latest-small\u.csv', sep='\t', names=column_names)


# In[5]:


df.head()


# In[7]:


movie_titles = pd.read_csv(r"A:\Ineuron\ml-latest-small\Movie_Id_Titles.csv")
movie_titles.head()


# In[8]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# In[10]:


#EDA- Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[12]:



df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[13]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# In[14]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# In[15]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# In[16]:



plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[17]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# In[18]:


#Recommending Similar Movies
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# In[19]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# In[20]:


ratings.head()


# In[21]:



starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


# In[22]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# In[23]:



corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# In[24]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[25]:



corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# In[26]:



corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# In[27]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# In[ ]:




