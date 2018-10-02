
# coding: utf-8

# In[1]:


from modeling_fns import *

import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hyperopt import fmin, Trials


# In[2]:


# Opening the cleaned data:
with open('to_model_top_4_classes.pickle', 'rb') as f:
    to_model = pickle.load(f)


# In[3]:


# train test split
X = to_model.drop(columns='Country.of.Origin')
y = to_model['Country.of.Origin']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)


# In[4]:


# set maximum number of evaluations for hyperopt: warning this can take a long time if you set it high
hyperopt_max_evals = 2


# In[5]:


#setting up and running hyperopt


# In[6]:


def validate_model(unfitted_model):
    '''
    Takes in an unfitted model, instantiated with desired parameters.
    Returns a tuple of cross-validated log loss and accuracy.
    Number of cv_folds is editable, default 8.
    '''
    logloss = -1 * cross_val_score(unfitted_model, X_train, y_train, cv=8, scoring='neg_log_loss').mean()
    acc = cross_val_score(unfitted_model, X_train, y_train, cv=8, scoring='accuracy').mean()
    return logloss, acc


# In[7]:


def score_fn(params):
    '''
    Scoring function for hyperopt.
    Takes in dict of params,
    Returns a dictionary of log loss, accuracy, and status (a hyperopt requirement)
    '''
    print(params)
    
    pipe = pipe_from_params(params)
    
    logloss, acc = validate_model(pipe)
    return_dict = {'loss': logloss, 'accuracy': acc, 'status':'ok'}
    print(return_dict)
    return return_dict


# In[8]:


def get_best_hyperparams(hyperopt_max_evals):
    '''
    takes in X_train and y_train.
    Returns best_model_choice (optimum hyperparameters) and model_choice_trials (hyperopt trials object)
    '''
    model_choice_trials = Trials()
    best_model_choice = fmin(score_fn,
                             space=opt_space,
                             algo=tpe.suggest,
                             max_evals=hyperopt_max_evals,
                             trials=model_choice_trials)
    return best_model_choice, model_choice_trials


# In[9]:


best_model_choice, model_choice_trials = get_best_hyperparams(hyperopt_max_evals)


# In[10]:


# save best hyperparameter information
with open('best_model_choice_test.pickle', 'wb') as f:
    pickle.dump(best_model_choice, f)


# In[11]:


# plot hyperopt trials
hyperopt_plot_file = 'hyperopt_trials.png'
plot_hyperopt_trials(model_choice_trials, hyperopt_plot_file)


# In[12]:


# implement best model as selected by hyperopt
best_model = get_best_pipeline(best_model_choice)
best_model.fit(X_train, y_train)


# In[13]:


# save best model
with open('best_model_trained.pickle', 'wb') as f:
    pickle.dump(best_model, f)


# In[14]:


class_dict = {'Brazil':0, 'Colombia':1, 'Guatemala':2, 'Mexico':3}


# In[15]:


# evaluating best model's performance
(test_acc, test_log_loss) = evaluate_best_model(best_model, X_test, y_test, class_dict)


# In[16]:


# plot test set confusion matrix with nice colors and labels
y_pred = best_model.predict(X_test)
labels=['Brazil','Colombia','Guatemala','Mexico']
confmx_filename = 'confusion_matrix.png'
plot_confmx(y_test,y_pred,labels, confmx_filename)


# In[17]:


# get feature importance estimation using permutation importance
feature_importances = perm_import(best_model, X_test, y_test, class_dict)
feature_importances


# In[18]:


# plot all data along feature dimensions (2 most important features)
feat_plot_filename= 'feat1_vs_feat2.png'
plot_data_by_feats('Balance', 'Flavor', to_model, class_dict, feat_plot_filename)


# In[19]:


# plot test set performance by feature value
perf_by_feat_filename = 'perform_by_feat.png'
model_performance_for_feature('Balance', X_test, y_test, y_pred, class_dict, perf_by_feat_filename)

