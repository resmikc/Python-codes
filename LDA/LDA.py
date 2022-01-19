#!/usr/bin/env python
# coding: utf-8

# ### Loading libraries

# In[13]:


#Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Load dataset

# In[2]:


#Load Dataset
dataset=pd.read_csv(r'D:\Data 2204\assignment instructions\WheatData.csv')
dataset.head()


# ### characteristics

# In[3]:


#Show Key Statistics
dataset.describe()


# ### Profile Report

# In[4]:


#Create Profile Report
    
#Importing package
import pandas_profiling as pp
from IPython.display import IFrame
    
#Profile Report
WheatDataReport = pp.ProfileReport(dataset)
WheatDataReport.to_file('WheatDataReportW5a.html')
display(IFrame('WheatDataReportW5a.html', width=900, height=350))


# ### covariance test

# In[5]:


#Covariance test - Levene Test

    #p <= alpha(0.05): reject H0, not the same covariance.
    #   p > alpha(0.05): fail to reject H0, same covariance.
    
import scipy.stats as stats
names=dataset.get('target')
    
cnt=1
for col in dataset.columns:
    if (col=='target'):
        continue
    
    stat, p = stats.levene(dataset[col][dataset['target'] == names[0]], 
    dataset[col][dataset['target'] == names[1]])

    print(col)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Covariance the Same (fail to reject H0)')
        print('\\n')
    else:
        print('Covariance different(reject H0)')
        print('\\n')
    cnt +=1  


# ### Normality test

# In[6]:


#Normality test - Shapiro-Wilk Test
# p <= alpha(0.05): reject H0, not normal.
# p > alpha(0.05): fail to reject H0, normal.
    
from scipy.stats import shapiro
    
cnt=1
for col in dataset.columns:
    if (col=='Outcome'):
        continue
    
    stat, p = shapiro(dataset[col])
    
    print(col)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
        print('\\n')
    else:
        print('Sample does not look Gaussian (reject H0)')
        print('\\n')
        cnt +=1    


# ### Data Preprocessing

# In[7]:


#Create x and y variables
x = dataset.drop('target', axis=1).to_numpy()
Y = dataset['target'].to_numpy()
    
#Create Train and Test Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,Y,test_size = 0.2,stratify=Y,random_state = 100)
    
#Fix the imbalanced Classes
from imblearn.over_sampling import SMOTE
smt=SMOTE(random_state=100)
x_train_smt,y_train_smt = smt.fit_resample(x_train,y_train)
    
#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train_smt)
x_test2 = sc.transform(x_test)
    
#Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ### Class Balance

# In[8]:


#Class Balance - Test Data
print('Train Data - Class Split')
num_zeros = (y_train_smt == 0).sum()
num_ones = (y_train_smt == 1).sum()
print('Class 0 -',  num_zeros)
print('Class 1 -',  num_ones)
print('class 2 Canadian -',num_ones)


# ### LDA model

# In[9]:


#Base LDA Model
from sklearn.metrics import classification_report, confusion_matrix  
    
for name,method in [('LDA', LinearDiscriminantAnalysis())]:
     
    method.fit(x_train2,y_train_smt)
    predict = method.predict(x_test2)
    print('\\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict)) 


# ### Pipelines

# In[10]:


#Construct some pipelines 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
    
#Create Pipeline
    
pipeline =[]
    
pipe_lda = Pipeline([('scl', StandardScaler()),
                    ('clf', LinearDiscriminantAnalysis())])
pipeline.insert(0,pipe_lda)
    
    
modelpara =[]
    
param_gridlda = {'clf__solver':['svd','lsqr','eigen']}
modelpara.insert(0,param_gridlda)


# ### Learning curve

# In[11]:


from sklearn.model_selection import learning_curve
    
def plot_learning_curves(model):
    train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                                            X=x_train_smt, 
                                                            y=y_train_smt,
                                                            train_sizes= np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            scoring='recall_weighted',random_state=100)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean,color='blue', marker='o', 
             markersize=5, label='training recall')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation recall')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid(True)
    plt.xlabel('Number of training samples')
    plt.ylabel('Recall')
    plt.legend(loc='best')
    plt.ylim([0.6, 1.0])
    plt.show()


# In[14]:


#Plot Learning Curve
print('LDA Learning Curve')
plot_learning_curves(pipe_lda)


# ### model analysis and boxplot

# In[15]:


#Model Analysis
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
    
models=[]
models.append(('LDA',pipe_lda))
#models.append(('QDA',pipe_qda))
#models.append(('Log Reg',pipe_logreg))
    
#Model Evaluation
results =[]
names=[]
scoring ='recall_weighted'
print('Model Evaluation - Recall Score')
for name, model in models:
    rkf=RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
    cv_results = cross_val_score(model,x,Y,cv=rkf,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('{} {:.2f} +/- {:.2f}'.format(name,cv_results.mean(),cv_results.std()))
print('\\n') 
    
fig = plt.figure(figsize=(10,5))
fig.suptitle('Boxplot View')
ax = fig.add_subplot(111)
sns.boxplot(data=results)
ax.set_xticklabels(names)
plt.ylabel('Recall')
plt.xlabel('Model')
plt.show()


# ### Optimized model

# In[16]:


#Define Gridsearch Function
    
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
    
def Gridsearch_cv(model, params):
    
    #Cross-validation Function
    cv2=RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
    
    #GridSearch CV
    gs_clf = GridSearchCV(model, params, cv=cv2,scoring='recall_weighted')
    gs_clf = gs_clf.fit(x_train_smt, y_train_smt)
    model = gs_clf.best_estimator_
     
    # Use best model and test data for final evaluation
    y_pred = model.predict(x_test)
    
    #Identify Best Parameters to Optimize the Model
    bestpara=str(gs_clf.best_params_)
    
    #Output Validation Statistics
    target_names=['Kama','Rosa','Canadian']
    print('\\nOptimized Model')
    print('\\nModel Name:',str(pipeline.named_steps['clf']))
    print('\\nBest Parameters:',bestpara)
    print('\\n', confusion_matrix(y_test,y_pred))  
    print('\\n',classification_report(y_test,y_pred,target_names=target_names)) 
     
    


# In[17]:


#Run Models
    
for pipeline, modelpara in zip(pipeline,modelpara):
    Gridsearch_cv(pipeline,modelpara)


# In[ ]:




