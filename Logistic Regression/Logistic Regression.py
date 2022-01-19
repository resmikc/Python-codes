#!/usr/bin/env python
# coding: utf-8

# ### load libraries

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### load dataset

# In[13]:


dataset = pd.read_csv(r'D:\Data 2204\assignment instructions\WheatData.csv')
dataset.head()
                      


# ### data characteristics

# In[3]:


#Show Key Statistics
dataset.describe()


# ### Class Balance

# In[4]:


#Class Balance\n",
print('Class Split')
print(dataset['target'].value_counts())
dataset['target'].value_counts().plot.bar(figsize=(10,4),title='Classes Split for Dataset')
plt.xlabel('Classes')
plt.ylabel('Count')


# ### data preprocessing

# In[14]:


#Create x and y variables
x = dataset.drop('target', axis=1).to_numpy()
Y = dataset['target'].to_numpy()

#Create Train and Test Datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.20,stratify=Y,random_state=100)
    
#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)
    
#Model
from sklearn.linear_model import LogisticRegression


# ### Logistic Regression model

# In[15]:


#Base Logistical Regression Model
from sklearn.metrics import classification_report, confusion_matrix  
    
for name,method in [('LogReg', LogisticRegression(multi_class='multinomial',solver='newton-cg',
                                             random_state=100,max_iter=10000))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    print('\\nEstimator: {}'.format(name))
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict))    


# ### Pipeline

# In[16]:


#Construct some pipelines 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
    
#Create Pipeline

pipeline =[]
pipe_logreg = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(multi_class='multinomial',solver='newton-cg',
                                               random_state=100,max_iter=10000))])
pipeline.insert(0,pipe_logreg)
    
# Set grid search params 

modelpara =[]
    
param_gridlogreg = {'clf__C': [0.01, 0.1, 1, 10, 100], 
                     'clf__penalty': ['l2'],
                    'clf__solver':['newton-cg', 'sag', 'saga' , 'lbfgs']}
modelpara.insert(0,param_gridlogreg)
   


# ### Learning curve

# In[8]:


#Define Plot for learning curve
    
from sklearn.model_selection import learning_curve
    
def plot_learning_curves(model):
    
    train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                                        X=x_train, 
                                                        y=y_train,
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
    plt.ylim([0.5, 1.01])
    plt.show()


# In[9]:


#Plot Learning Curve
print('Mulit-Logisistic Regression - Learning Curve')
plot_learning_curves(pipe_logreg)


# ### Bias and Variance

# In[10]:


#Bias and Variance Trade-Off
from mlxtend.evaluate import bias_variance_decomp
    
#Convert Object columns to numeric
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
ytestConv = encoder.fit_transform(y_test)
ytrainConv = encoder.fit_transform(y_train)

#Bias Variance Script
print('Bias Variance Trade-Off')
for name,method in[('Multi-Logistical Model',pipe_logreg)]:        
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        method, x_train2, ytrainConv, x_test2, ytestConv, 
        loss='0-1_loss',
        random_seed=100)
    print('\\nEstimator: {}'.format(name)) 
    print('\\nAverage Bias: {:.2f}'.format(avg_bias))
    print('Average Variance: {:.2f}'.format(avg_var))      
   


# ### Model evaluation

# In[11]:


#Model Analysis
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
    
models=[]
models.append(('Logistic Regression',pipe_logreg))
    
#Model Evaluation
results =[]
names=[]
scoring ='accuracy'
print('Model Evaluation - Accuracy Score')
for name, model in models:
    rkf=RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
    cv_results = cross_val_score(model,x,Y,cv=rkf,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('{} {:.2f} +/- {:.2f}'.format(name,cv_results.mean(),cv_results.std()))
print('\\n')

fig = plt.figure(figsize=(5,5))
fig.suptitle('Boxplot View')
ax = fig.add_subplot(111)
sns.boxplot(data=results)
ax.set_xticklabels(names)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()


# ### Optimized model

# In[17]:


#Define Gridsearch Function
    
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
    
def Gridsearch_cv(model, params):
    
    #Cross-validation Function
    cv2=RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

    #GridSearch CV
    gs_clf = GridSearchCV(model, params, cv=cv2,scoring='accuracy')
    gs_clf = gs_clf.fit(x_train, y_train)
    model = gs_clf.best_estimator_
    
    # Use best model and test data for final evaluation
    y_pred = model.predict(x_test)
    
    #Identify Best Parameters to Optimize the Model
    bestpara=str(gs_clf.best_params_)
    
    #Output Heading
    print('\\nOptimized Model')
    print('\\nModel Name:',str(pipeline.named_steps['clf']))
    
    #Output Validation Statistics
    target_names=['Kama','Rosa','Canadian']
    print('\\nBest Parameters:',bestpara)
    print('\\n', confusion_matrix(y_test,y_pred)) 
    print('\\n',classification_report(y_test,y_pred,target_names=target_names))        
   


# In[18]:


#Run Models
    
for pipeline, modelpara in zip(pipeline,modelpara):
    Gridsearch_cv(pipeline,modelpara)


# ### ROC Curve

# In[31]:


#Creating Multivariable ROC Curve
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
    
#Binarize the output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
le = LabelEncoder()
y2=dataset['target']
y3= le.fit_transform(y2)
y3 = label_binarize(y3, classes=[0, 1, 2])
x2 = dataset.drop('target', axis=1).values
    
#Creating Training and Testing Data
x_train3, x_test3, y_train3, y_test3 = train_test_split(x2, y3, test_size=0.20,stratify=Y,random_state=100)
n_classes = y3.shape[1]
    
#Create ROC Curve
from sklearn.multiclass import OneVsRestClassifier
classOrg = model
classifier = OneVsRestClassifier(model)
y_score = classifier.fit(x_train3, y_train3).predict_proba(x_test3)
    
#Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test3[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])  
    
#Compute macro-average ROC curve and ROC area
    
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] =all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
#Plot all ROC curves
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
               label='ROC curve of class {0} (area = {1:0.2f})'
               ''.format(i, roc_auc[i]))
    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()   


# ### Feature Selection

# In[32]:


#Next Steps - Feature Selection using SelectFromModel
from sklearn.feature_selection import SelectFromModel
clf = LogisticRegression(multi_class='multinomial',solver='newton-cg',
                                               random_state=100,max_iter=10000)
clf.fit(x_train2,y_train)
model2 = SelectFromModel(clf, prefit=True)
feature_idx = model2.get_support()
feature_name = dataset.drop('target',axis=1).columns[feature_idx]
print('\\nKey Features:',feature_name)


# In[ ]:




