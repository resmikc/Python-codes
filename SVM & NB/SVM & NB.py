#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[23]:


#Importing the required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# ### Importing dataset

# In[24]:


#Load Data
df = pd.read_csv(r'C:\Users\resmi\Downloads\cancer.csv')
df


# In[5]:


#Using pandas "shape" function to know the number of attributes in the dataset

df.shape


# ### datatypes

# In[6]:


#Using pandas "dtypes" function to get the data type of all columns in the dataframe

df.dtypes


# ### missing values

# In[7]:


#checking missing values in the dataset

df.isnull().sum()


# ### key characteristics

# In[8]:


#Overall Statistics of the Dataset

df.describe()


# ### correlation

# In[9]:


#Using pandas "corr" method to know correlation among variables in the dataset

dfcorr = df.corr()
dfcorr


# ### correlation visualization

# In[10]:


#Correlation Visualization

plt.figure(figsize=(14,8))
sns.heatmap(dfcorr, annot = True, cmap = "Blues")


# ### histogram

# In[11]:


#Using matplotlib's "hist" function to visualize distribution of data for each column in the dataset

df.hist(figsize = (12,11), grid= False, color = "blue")
plt.show()


# In[25]:


#Using seaborn's "boxplot" function to generate boxplot for each column in the dataset in order to find outliers

plt.figure(figsize =(16,6))
sns.boxplot(data= df)


# ### data preprocessing

# In[26]:


#Defining the variables

#defining "x" or independent variables in the dataset

x = df.drop("Class", axis = 1).to_numpy()  #using pandas "drop" attribute to drop Class column from x as it is a target variable 

#placing target variable "Class" in y variable

y = df["Class"].to_numpy() #Assigning y variable to "Class" column as it is the dependent variable in the dataset 

#Randomly splitting data into training and testing data using function train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y, test_size=0.2,random_state=100)  

#the test_size parameter sets the proportion of data that is split into the testing set. 
#Thus, here 20% of data samples will be utilized for testing

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[27]:


#Using "StandardScaler" to standardize the data

sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# ### SVM & NB

# In[28]:


#Using sklearn's svm and naive_bayes libraries to create svm and naive bayes script

for name,method in [('SVM', SVC(kernel='linear',random_state=100)),
                    ('Naive Bayes',GaussianNB())]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict))   


# In[ ]:




