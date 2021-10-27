#!/usr/bin/env python
# coding: utf-8

# In[116]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[117]:


df=pd.read_csv(r'C:\Users\new\Documents\PythonFiles\Kaggle Practice\house-prices-advanced-regression-techniques\train.csv')


# In[118]:


df.head()


# In[119]:


df.isnull().sum()


# In[120]:


df.info()


# In[121]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:





# In[122]:


df['MSZoning'].value_counts()


# In[123]:


df.shape


# In[124]:


## Fill Missing Values

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[125]:


df.drop(['Alley'],axis=1,inplace=True)
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[126]:


df.shape


# In[127]:


df.drop(['Id'],axis=1,inplace=True)


# In[128]:


df.shape


# In[129]:


obj = df.isnull().sum().sort_values(ascending=False)
for key,value in obj.iteritems():
    print(key,"-",value)


# In[130]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[131]:


df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[132]:


df.isnull().sum().sort_values(ascending=False)


# In[133]:


df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])


# In[134]:


df.isnull().sum().sort_values(ascending=False)


# In[135]:


df.shape


# In[136]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[137]:


df.head(5)


# In[138]:


##HAndle Categorical Features


# In[139]:


columns=df.select_dtypes(include=['object', 'category']).columns


# In[140]:


len(columns)


# In[141]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[142]:


main_df=df.copy()


# In[143]:


## Combine Test Data 

test_df=pd.read_csv(r'C:\Users\new\Documents\PythonFiles\Kaggle Practice\house-prices-advanced-regression-techniques\formulatedtest.csv')


# In[144]:


test_df.shape


# In[145]:


df.shape


# In[146]:


final_df=pd.concat([df,test_df],axis=0)


# In[147]:


final_df.shape


# In[148]:


final_df['SalePrice']


# In[149]:


final_df=category_onehot_multcols(columns)


# In[150]:


final_df.shape


# In[151]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[153]:


final_df.shape


# In[154]:


df_Train=final_df.iloc[:1460,:]
df_Test=final_df.iloc[1460:,:]


# In[157]:


df_Train.head()


# In[158]:


df_Test.head()


# In[159]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[160]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[161]:


df_Train.shape


# # Predicting Algorithms

# In[162]:


import xgboost
classifier=xgboost.XGBRegressor()
from sklearn.model_selection import RandomizedSearchCV


# ## Hyper Parameter Optimization

# In[163]:


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]


# ### Define the grid of hyperparameters to search

# In[164]:


hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# ### Set up the random search with 4-fold cross validation

# In[ ]:





# In[165]:


random_cv = RandomizedSearchCV(estimator=classifier,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[166]:


random_cv.fit(X_train,y_train)


# In[167]:


random_cv.best_estimator_


# In[169]:


regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=900, n_jobs=4, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)


# In[170]:


regressor.fit(X_train,y_train)


# In[171]:


import pickle
filename = r'C:\Users\new\Documents\PythonFiles\Kaggle Practice\house-prices-advanced-regression-techniques\finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[172]:


df_Test.shape


# In[175]:



df_Test.head()


# In[177]:


y_pred=regressor.predict(df_Test)


# In[178]:


y_pred


# ## Create Sample Submission file and Submit

# In[181]:


pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv(r'C:\Users\new\Documents\PythonFiles\Kaggle Practice\house-prices-advanced-regression-techniques\sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)


# In[194]:


pred.columns=['SalePrice']


# In[195]:


temp_df=df_Train['SalePrice'].copy()


# In[196]:



temp_df.column=['SalePrice']


# In[197]:


df_Train.drop(['SalePrice'],axis=1,inplace=True)


# In[198]:



df_Train=pd.concat([df_Train,temp_df],axis=1)


# In[199]:



df_Test.head()


# In[200]:



df_Test=pd.concat([df_Test,pred],axis=1)


# In[201]:


df_Train=pd.concat([df_Train,df_Test],axis=0)


# In[202]:


df_Train.shape


# In[203]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# # Importing the Keras libraries and packages

# In[204]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[205]:


from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[207]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 50, activation='relu',input_dim = 174))

# Adding the second hidden layer
classifier.add(Dense(units = 25, activation='relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 50, activation='relu'))
# Adding the output layer
classifier.add(Dense(units = 1,))

# Compiling the ANN
classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, num_epoch = 1000)


# In[ ]:




