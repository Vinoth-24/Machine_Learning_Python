#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import string
from pandas import Series,ExcelWriter

#reading data and preprocessing data.
overall_data= pd.read_excel(r"C:\Users\new\Documents\PythonFiles\Trell project\Work\Input sheet\Trail_Performance_Malayalam_2021_09_24.xlsx",sheet_name="Overall")
cat=pd.read_excel(r"C:\Users\new\Documents\PythonFiles\Trell project\Work\Input sheet\Trail_Performance_Malayalam_2021_09_24.xlsx",sheet_name="Final Ranking")
overall_data['CP_0sec']= overall_data['CP_0sec']*100
data = overall_data[overall_data['CP_0sec'].notna()]
overall_data['CP_0sec']= overall_data['CP_0sec'].map("{:,.2f}%".format)

# number of categories as input.
categories = [] 
n = int(input("Enter number of categories : "))
 
# iterating till all the categories is input.
#Note: Category names should be exactly matching the L1 category names.
for i in range(0, n):
    ele = str(input())
   # if ele == "beard":
   #     ele= ["Beard Care", "Beard Styles"]
    categories.append(ele)
print(categories)

# sorting algorithm
null=list()
z=0
null.insert(z,cat)
z=z+1
null.insert(z,overall_data)
for i in categories:
    z=z+1
    #if i == "Beard":
     #   i = ['Beard Care', 'Beard Styles']
        #data_1=data[data.L1_category.isin(i)]
    #else:
    data_1 = data[data['L1_category'].str.contains(i, na=False)]
    a = data_1.sort_values(by ='views_0sec', ascending = 0)
    data_2 = a[a['views_0sec'] > a['views_0sec'].mean()]
    b = data_2.sort_values(by ='CP_0sec', ascending = 0)
    data_3 = b[b['CP_0sec'] > 15.00]
    c = data_3.sort_values(by ='average_timespent', ascending = 0)
    data_4 = b[b['CP_0sec'] < 15.00]
    d = data_4.sort_values(by ='average_timespent', ascending = 0)
    c_d = pd.concat([c, d], axis=0)
    data_5 = a[a['views_0sec'] < a['views_0sec'].mean()]
    e = data_5.sort_values(by ='CP_0sec', ascending = 0)
    data_6 = e[e['CP_0sec'] > 15.00]
    f = data_6.sort_values(by ='average_timespent', ascending = 0)
    data_7 = e[e['CP_0sec'] < 15.00]
    g = data_7.sort_values(by ='average_timespent', ascending = 0)
    f_g = pd.concat([f, g], axis=0)
    c_d_f_g = pd.concat([c_d, f_g], axis=0)
    Final_data= c_d_f_g
    Final_data['CP_0sec'] = Final_data['CP_0sec'].map("{:,.2f}%".format)
    null.insert(z,Final_data)
    
# replacing unwanted symbols for Sheetnames    
p=0
for i in categories:
    categories[p] = i.replace(r'/', ' ')
    
    p=p+1
    
# writing data sheet by sheet onto Excel Workbook
with pd.ExcelWriter('C:\\Users\\new\Documents\\PythonFiles\\Trell project\\Work\\Output sheet\\Trail_Performance_Malayalam_24.09.21.xlsx') as writer:
    j=0
    null[j].to_excel(writer, sheet_name= 'Categories', index=False)
    j=j+1
    null[j].to_excel(writer, sheet_name= 'Overall Data', index=False)
    for i in range(len(categories)):
        sheetname=categories[i]
        j=j+1
        null[j].to_excel(writer, sheet_name= sheetname, index=False)
        


# In[2]:


df[df['A'].str.contains("hello")]

