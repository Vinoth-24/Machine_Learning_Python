#!/usr/bin/env python
# coding: utf-8

# In[102]:


# Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import numpy
import pandas as pd


# In[103]:


# fixing random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[104]:


# Load Dataset
dataset = pd.read_csv(r"A:\Data Science by Excelr\Data science assignments\Neural Networks\forestfires.csv")
dataset.head(10)


# In[105]:


dataset.shape


# In[106]:


dataset.dtypes


# In[107]:


dataset.info()


# In[108]:


dataset.describe()


# In[ ]:





# In[109]:


# Label Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataset["month"] = label_encoder.fit_transform(dataset["month"])
dataset["day"] = label_encoder.fit_transform(dataset["day"])
dataset["size_category"] = label_encoder.fit_transform(dataset["size_category"])


# In[110]:


dataset.head(5)


# In[111]:


# split into input (X) and output (Y) variables
X = dataset.iloc[:,:11]


# In[112]:


Y = dataset.iloc[:,-1]


# In[113]:


X


# In[114]:


Y


# In[115]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[116]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[117]:


X_train


# In[ ]:





# In[118]:


# Build ANN Model
# create model
model = Sequential()
model.add(layers.Dense(50, input_dim=11,  activation='relu'))
model.add(layers.Dense(11,  activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[119]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[120]:


# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10)


# In[121]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[122]:


# Visualization
history.history.keys()


# In[123]:


# summarize history for accuracy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[124]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




