#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


#Importing "Concrete_Data" dataset 

df = pd.read_csv(r'C:\Users\resmi\Downloads\Concrete_Data.csv')
df


# ### Checking the datatype

# In[3]:


#Using pandas "dtypes" function to get data type of all columns

df.dtypes


# ### Finding the missing values

# In[4]:


df.isnull().sum()


# ### Key statistics

# In[5]:


#Using pandas "describe" attribute to get overall statistical summary of the given dataframe

df.describe()


# ### Correlation

# In[6]:


#Using pandas "corr" function to check the correlation among variables

dfcorr = df.corr()
dfcorr


# ### correlation visualization

# In[7]:


#Using seaborn's "heatmap" to visualize the correlation among variables 

plt.figure(figsize=(14,8))
sns.heatmap(dfcorr, annot = True, cmap = "Blues")


# ### Data Preprocessing

# In[8]:


#Defining the variables

#defining "x" or independent variables

x = df.drop("CMS", axis = 1).to_numpy()  #using pandas "drop" attribute to drop CMS column as it is a target variable from x

#placing target variable "CMS" in y variable

y = df["CMS"].to_numpy()


# In[9]:


#Randomly splitting data into training and testing data using function train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)  

#the test_size parameter sets the proportion of data that is split into the testing set. 
#Thus, here 20% of data samples will be utilized for testing

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[10]:


#Using "StandardScaler" to standardize the data

sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# ### model building

# In[11]:


#Multiple Regression script to predict CMS(Concrete compressive strength)

for name,method in [('Linear Regression', LinearRegression(n_jobs=-1))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)

print('Method: {}'.format(name))   

#Coefficents
print('\nIntercept: {:0.2f}'.format(float(method.intercept_)))
coeff_table=pd.DataFrame(np.transpose(method.coef_),df.drop('CMS',axis=1).columns,columns=['Coefficients'])
print('\n')
print(coeff_table)
    
#Getting R^2, mean absolute error, Mean squared error and root mean squared error

print('\nR2: {:0.2f}'.format(metrics.r2_score(y_test, predict)))
print('Mean Absolute Error: {:0.2f}'.format(metrics.mean_absolute_error(y_test, predict)))  
print('Mean Squared Error: {:0.2f}'.format(metrics.mean_squared_error(y_test, predict)))  
print('Root Mean Squared Error: {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, predict))))


# ### comparing original with predicted values

# In[12]:


#Creating a dataframe for predicted values and orginal values

predict2 = predict.T
diff = predict2-y_test
predicted_df = {'Original Values':y_test,'Predicted Values':predict2.round(1),'Difference':diff.round(1)}

pd.DataFrame(predicted_df).head(15)


# In[ ]:




