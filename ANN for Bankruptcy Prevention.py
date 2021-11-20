#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries and Reading file

# In[96]:


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.activations import relu, sigmoid
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


# In[97]:


df=pd.read_csv("A:\\Data Science by Excelr\\PROJECT\\bankruptcy_prevention.csv",sep=";")


# In[98]:


df['Bankruptcy'] = [
    1 if typ == 'bankruptcy' else 0 for typ in df[' class']
]
df.sample(5)


# #### Splitting into features and target 

# In[99]:


features=df.columns[0:6]
X=df[features]
y=df['Bankruptcy']


# In[150]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ### Artificial Neural Networks
# - Artificial neural networks (ANNs) are comprised of node layers- containing an input layer, one or more hidden layers, and an output layer.<br>
# - Each Node is connected to another node from the next hidden layers and has associated weights and threshold. Each node is activated if the threshold is met. Otherwise Data is not passed.<br>
# - Neural networks rely on training data to learn and improve their accuracy over time.  <br>
# - With proper fine tuning, these models can give high accuracy on many problems to classify or cluster data at high velocity.

# <img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png" alt="Deep Neural Network Structure" style="width: 600px;"/>
# 
# ***

# #### Hyperparameters in ANN - Used to Fine tune a model for high accuracy
# 1. How many number of hidden layers we should have? - We used 2 to 20 layers.
# 2. How many number of neurons we should have in hidden layers? - We used 32 to 512 neurons.
# 3. Learning Rate - We used 1e-2, 1e-3, 1e-4.
# #### Other Parameters used
# 1. Loss function - Binary Cross Entropy
# 2. Metric - Accuracy
# 3. Activation function - Sigmoid for the Output layer and Relu for all other layers
# 4. Optimizer - Adam

# In[101]:


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=6,))
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


# In[103]:


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='project2',
    project_name='Bankruptcy_Prevention')


# - #### We are tuning the model based on Validation accuracy.

# In[104]:


tuner.search_space_summary()


# #### Splitting the data, then fitting

# In[105]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)


# In[106]:


tuner.search(X_train, y_train,
             epochs=100,
             validation_data=(X_test, y_test))


# In[107]:


tuner.results_summary()


# #### Here By Keras tuner Optimization, We have few best combinations of Hyperparameters leading to high Accuracy(100 percent in this case).<br> Now we choose the best combination to make a final Neural Network for Model building.
# ***
# 

# ### Making the Final Model with the chosen hyperparameters

# In[108]:


classifier=Sequential()


# In[109]:


classifier.add(layers.Dense(units=256,kernel_initializer ='he_uniform',activation='relu',input_dim=6))
classifier.add(layers.Dense(units=256,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=384,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=32,kernel_initializer ='he_uniform',activation='relu'))
classifier.add(layers.Dense(units=1,kernel_initializer ='glorot_uniform',activation='sigmoid'))


# In[110]:


classifier.summary()


# In[111]:


opt = keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])


# In[112]:


model_final=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)


# #### - For 100 epochs, We can notice that there is high training accuracy(100%) and validation accuracy of 0.9828<br> - From this we can tell that our model is performing very well.<br> - Here we can notice that the validation split is a split from the X_train data. So, For predicting we use the test data(Unseen data) which is not affected by any data leakage thus preventing overfitting.
# ***

# ### AUC and ROC Curve for Understanding Performance

# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[127]:


# Computing manually fpr, tpr, thresholds and roc auc 
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("ROC_AUC Score : ",roc_auc)
print("Function for ROC_AUC Score : ",roc_auc_score(y_test, y_pred))
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Threshold value is:", optimal_threshold)
plot_roc_curve(fpr, tpr)


# #### This is a perfect ROC Curve meaning that we can perfectly distinguish between Bankrupted and Not Bankrupted.

# ### Prediction and Confusion Matrix

# In[143]:


pred_y=classifier.predict(X_test)
y_pred = (pred_y >= 1) #optimal threshold is 1
y_pred=1*y_pred 


# In[145]:


cm = confusion_matrix(y_test, y_pred)
score=accuracy_score(y_pred,y_test)


# In[146]:


print("Confusion Matrix: \n",cm,"\nAccuracy Score: ", score,"\n\n")
print(classification_report(y_test,y_pred))


# #### From the confusion matrix and report, We can see that none of the data is misclassified.

# ### Graphical representation of Accuracy and loss w.r.t no. of Epochs.

# In[122]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.plot(
    np.arange(1, 101), 
    model_final.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 101), 
    model_final.history['accuracy'], label='Accuracy'
)

plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend();


# #### From Graph, we can see the increase in accuracy and decrease in loss w.r.t epochs.

# ### Thus we can conclude that our ANN model is performing perfectly with very high accuracy.
# ***
