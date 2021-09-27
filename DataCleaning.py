#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data= pd.read_csv("C:\\Users\\new\\Documents\\PythonFiles\\Github practice\\data_cleaning_challenge.csv")


# In[ ]:


data


# In[ ]:


drop=data.drop(columns=["Unnamed: 9","Unnamed: 10"])


# In[ ]:


drop


# In[ ]:


drop_Nan= drop[drop["Row Type"].notna()]


# In[ ]:


drop_Nan


# In[ ]:


column_values=[]
counter=0

for i in drop_Nan["Row Type"]:
    if "first name" in i:
        counter+=1
    column_values.append(counter)
        


# In[ ]:


(column_values)


# In[ ]:


iter_cols=drop_Nan
iter_cols


# In[ ]:


iter_cols["Iteration"]=column_values
iter_cols


# In[ ]:


drop_extra_column_names=iter_cols[iter_cols["Row Type"] != "Row Type"]


# In[ ]:


drop_extra_column_names


# In[ ]:


Name_data=drop_extra_column_names[drop_extra_column_names["Row Type"].str.contains("first name")]


# In[ ]:


Name_data.drop(columns=["Speed1","Speed2","Electricity","Effort","Weight","Torque"],inplace=True)
Name_data.rename(columns={"Row Type":"First Name","Iter Number": "Last Name", "Power1":"Date"},inplace=True)
Name_data["First Name"]= Name_data["First Name"].str[12:]
Name_data["Last Name"]= Name_data["Last Name"].str[11:]
Name_data["Date"]= Name_data["Date"].str[6:]
Name_data


# In[ ]:


Noname_data=drop_extra_column_names[~drop_extra_column_names["Row Type"].str.contains("first name")]


# In[ ]:


Noname_data


# In[ ]:


pd.merge(left=Name_data,right=Noname_data,how="inner",on="Iteration")


# In[ ]:




