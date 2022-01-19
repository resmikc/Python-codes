#!/usr/bin/env python
# coding: utf-8

# ### Loading libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### load dataset

# In[3]:


dataset = pd.read_csv(r'D:\Data 2204\assignment instructions\EnergyUse-Heating.csv')
dataset.head()


# ### dataset characteristics

# In[3]:


dataset.info()


# ### checking missing values

# In[4]:


dataset.isnull().sum()


# ### key characteristics

# In[5]:


dataset.describe()


# In[4]:


dataset.corr()


# ### correlation visualization

# In[5]:


fig = plt.figure(figsize=(10,5))
sns.heatmap(dataset.corr(),annot=True,cmap="Blues")


# ### find the independent column correlations

# In[7]:


def correlation(dataset,threshold):
    col_corr= [] # List of correlated columns
    corr_matrix=dataset.corr() #finding correlation between columns
    for i in range (len(corr_matrix.columns)): #Number of columns
        for j in range (i):
            if abs(corr_matrix.iloc[i,j])>threshold:#checking correlation between columns\n"
                colName=(corr_matrix.columns[i], corr_matrix.columns[j]) 
                #getting correlated columns\n",
                col_corr.append(colName) #adding correlated column name\n",
    return col_corr #returning set of column names\n",
col=correlation(dataset,0.8)
print('Correlated columns @ 0.8:', col)
   


# ### data preprocessing

# In[8]:


x = dataset.drop('Y', axis=1).to_numpy()
Y = dataset['Y'].to_numpy()

#Create x and y datasets

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test = train_test_split(x,Y,test_size = 0.2,random_state = 100) 
    
#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)
    
#Model
from sklearn.neighbors import KNeighborsRegressor


# ### Learning Curve

# In[9]:


#Learning Curve
from sklearn.model_selection import learning_curve
    
def plot_learning_curves(model):
    train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                                            X=x_train2,
                                                            y=Y_train,
                                                            train_sizes=np.linspace(.1,1,10),
                                                            scoring = 'neg_root_mean_squared_error',
                                                            cv=10, random_state=100)
                                                                                                                       
                                                          
    train_mean = np.sqrt(np.mean(-train_scores, axis=1))
    train_std = np.sqrt(np.std(-train_scores, axis=1))
    test_mean = np.sqrt(np.mean(-test_scores, axis=1))
    test_std = np.sqrt(np.std(-test_scores, axis=1))
    
    
    plt.plot(train_sizes, train_mean,color='blue', marker='o',
         markersize=5, label='training accuracy')
    
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid(True)
    plt.xlabel('Number of training samples')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.ylim([0,10])
    plt.show()


# In[10]:


#Plot Learning Curve
print('k-NN Regressor Learning Curve')
plot_learning_curves(KNeighborsRegressor(2))


# ### original model 

# In[11]:


#Original Model - KNN

#Optimal value of K for KNN
from sklearn.model_selection import cross_val_score
from sklearn import metrics  
k_range = range(1, 10)
k_scores = []
    
for k in k_range:
    knn_org = KNeighborsRegressor(n_neighbors=k)
#obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn_org, x_train2, Y_train, cv=10, scoring='neg_root_mean_squared_error')
#append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())
    
#Print Best Score
BestScore = [1 - x for x in k_scores]
best_k = k_range[BestScore.index(min(BestScore))]

#Create Orginal KNN model
classifier_org = KNeighborsRegressor(n_neighbors = best_k)
    
#Fit KNN Model
classifier_org.fit(x_train2, Y_train)
    
#Prediction
y_pred_org = classifier_org.predict(x_test2)
    
print('Original Model')
print('\\nn_neighbors:',str(best_k))
print('\\nR2: {:.2f}'.format(metrics.r2_score(Y_test, y_pred_org)))
adjusted_r_squared = 1-(1-metrics.r2_score(Y_test,y_pred_org))*(len(Y)-1)/(len(Y)-x.shape[1]-1)
print('Adj_R2: {:0.2f}'.format(adjusted_r_squared))
print('Mean Absolute Error: {:0.2f}'.format(metrics.mean_absolute_error(Y_test, y_pred_org)))  
print('Mean Squared Error: {:0.2f}'.format(metrics.mean_squared_error(Y_test, y_pred_org)))  
print('Root Mean Squared Error: {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(Y_test, y_pred_org)))) 


# ### Optimized model

# In[12]:


#Gridsearch
    
from sklearn.model_selection import GridSearchCV
    
#k-NN Regression Model
knnreg2 = KNeighborsRegressor()
k_range = range(1, 10)
param_grid = { 
            'n_neighbors': k_range,
            'algorithm' : ['auto','ball_tree','kd_tree','brute'],
            'weights' : ['uniform','distance']}
    
knn_model = GridSearchCV(knnreg2, param_grid, cv=10, verbose=0,
                            scoring='neg_root_mean_squared_error')
    
grids = [knn_model] 
grid_dict = {0:'k-NN Regression Model'}
    
#Model Creation
    
#Create Heading
print('Optimized Model') 
    
#Fit the grid search objects 
for idx, optmodel in enumerate(grids): 
    print('\\nEstimator: {}'.format(grid_dict[idx])) 
    #Fit grid search
    optmodel.fit(x_train2, Y_train) 
    #Best params 
    print('\\nBest params: {}'.format(optmodel.best_params_)) 
    # Predict on test data with best params 
    y_pred3 = optmodel.predict(x_test2) 
    # Test data accuracy of model with best params    
    print('\\nR2: {:.2f}'.format(metrics.r2_score(Y_test, y_pred3)))
    adjusted_r_squared = 1-(1-metrics.r2_score(Y_test,y_pred3))*(len(Y)-1)/(len(Y)-x.shape[1]-1)
    print('Adj_R2: {:0.2f}'.format(adjusted_r_squared))  
    #Print MSE and RMSE\n",
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(Y_test, y_pred3)))
    print('Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(Y_test, y_pred3)))
    print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(Y_test, y_pred3))))


# In[ ]:




