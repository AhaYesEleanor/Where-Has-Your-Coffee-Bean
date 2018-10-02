
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import ExpSineSquared

from hyperopt import fmin, hp, tpe, Trials

from copy import deepcopy


# In[13]:


def pipe_from_params(params):
    '''
    Function used to simplify hyperopt score_fn implementation.
    Takes in a dictionary of model parameters (including model choice).
    Returns an unfitted pipeline.
    '''
    model_params = deepcopy(params)
    
    for key in params.keys():
        if type(model_params[key]) == dict:
            sub_dict = model_params.pop(key)
            model_params.update(sub_dict)
    
    model_type = model_params.pop('type')
    
    if model_type == 'rfc':
        model_params['max_depth'] = int(model_params['max_depth'])
        model_params['n_estimators'] = int(model_params['n_estimators'])
        model = RandomForestClassifier(n_jobs=3,
                                       **model_params)
    elif model_type == 'svc':
        model = SVC(probability=True,
                   **model_params)
        
    elif model_type == 'sgd':
        model = SGDClassifier(loss='log',
                              n_jobs=3,
                              **model_params)
        
    elif model_type == 'gpc':
        model = GaussianProcessClassifier(RBF(model_params['length_scale']))
        
    elif model_type == 'dtc':
        model_params['max_depth'] = int(model_params['max_depth'])
        model = DecisionTreeClassifier(**model_params)
    
    pipe = make_pipeline(StandardScaler(), model)
    return pipe


# In[26]:


# define categorical options for hyperparameters
class_weight_options = (None,"balanced")
svc_kernel_options = ('linear','rbf', 'poly', 'sigmoid')
svc_kernel_dicts = [{'kernel': 'linear'}, ]
sgd_penalty_options = ('l1', 'l2', 'elasticnet')
criterion_options = ('gini', 'entropy')
model_options = ('rfc', 'svc', 'sgd', 'gpc', 'dtc')


# In[27]:


opt_space = hp.choice('model_type',
                      [{'type': 'rfc',
                        'max_depth': hp.quniform('rfc_max_depth', 2, 10, 1),
                        'class_weight': hp.choice('rfc_class_wt', class_weight_options),
                        'n_estimators': hp.quniform('rfc_n_ests', 5, 75, 5),
                        'criterion': hp.choice('rfc_criterion', criterion_options),
                        'min_samples_leaf': hp.uniform('rfc_min_samples', 0.05, 0.5),
                       },
                       {'type': 'svc',
                        'C': hp.lognormal('svc_C', 0, 1),
                        'class_weight': hp.choice('svc_class_wt', class_weight_options),
                        'kernel': hp.choice('svc_kernel', 
                                            [{'kernel': 'linear'},
                                             {'kernel': 'rbf',
                                              'gamma': hp.uniform('svc_gamma_rbf', 0.01, 1)},
                                             {'kernel': 'poly',
                                              'degree': hp.quniform('svc_degree', 2, 5, 1),
                                              'gamma': hp.uniform('svc_gamma_poly', 0.01, 1)},
                                             {'kernel': 'sigmoid',
                                              'gamma': hp.uniform('svc_gamma_sigmoid', 0.01, 1)}
                                            ])   
                       },
                       {'type': 'sgd',
                        'alpha': hp.uniform('sgd_alpha', 0.01, 1),
                        'penalty': hp.choice('sgd_penalty', sgd_penalty_options),
                        'l1_ratio': hp.uniform('sgd_ratio', 0, 1)
                       },
                       {'type': 'gpc',
                        'length_scale': hp.uniform('gpc_lscale', 0.05, 7)
                       },
                       #{'type': 'gnb'},
                       {'type': 'dtc',
                        'criterion': hp.choice('dtc_criterion', criterion_options),
                        'max_depth': hp.quniform('dtc_max_depth', 3, 20, 1),
                        'min_samples_leaf': hp.uniform('dtc_min_samples', 0.05, 0.5),
                        'class_weight': hp.choice('dtc_class_wt', class_weight_options),
                       }
                      ])


# In[ ]:


def evaluate_best_model(best_model, X_test, y_test, class_dict):
    '''
    Takes in fitted model, test data, and dict of classes.
    Returns accuracy and log loss for the model on the test set.
    '''
    y_pred = best_model.predict(X_test)
    y_pred_t = pd.get_dummies(pd.Series(y_pred),)
    y_test_t = pd.Series(y_test).apply(lambda x: class_dict[x]).values

    accuracy = accuracy_score(y_test, y_pred)

    y_pred_prob = best_model.predict_proba(X_test)
    y_test_n = [class_dict[country] for country in y_test]
    y_test_n = pd.get_dummies(y_test_n)

    loss = log_loss(y_test_n, y_pred_prob)

    print('Test set accuracy: {} \n Test set log loss: {}'.format(accuracy, loss))
    return (accuracy, loss)


# In[ ]:


def plot_hyperopt_trials(model_choice_trials, hyperopt_plot_file):
    '''
    Plots hyperopt trials with log loss as y axis, trial number as x axis
    Takes in trials object populated by running hyperopt's fmin
    Colors have been selected for colorblind readability as well as beauty
    '''
    losses = model_choice_trials.losses()
    trials, model_types = zip(*((trial_n, trial_dict['misc']['vals']['model_type']) for trial_n, trial_dict in enumerate(model_choice_trials.trials)))
    
    x = list(trials)
    y = list(losses)
    
    classifiers = ('Random Forest', 'Support Vector Machine', 'Logistic Regression', 
                   'Gaussian Process', 'Decision Tree')
    label = list((num_list[0] for num_list in model_types))
    colors = ['#d98f5e', '#664C3e', '#9cd81e', '#6eb6be', '#9b98fd']
    cmap = matplotlib.colors.ListedColormap(colors)
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#fffbf0')

    plt.scatter(x, y, c=label, cmap=cmap, alpha=0.8)
    plt.yscale('log')
    plt.ylim(1.05,1.5)
    plt.yticks([1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.3862 , 1.4, 1.45],
               ['1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', 'Random', '1.4', '1.45'])
    plt.ylabel('Log Loss')
    plt.xlabel('Trials')
    
    c = [mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor=colors[i], edgecolor="none") for i in range(len(classifiers))]
    plt.legend(c,classifiers,loc=(1.01,0.8), facecolor='none', edgecolor='none')

    plt.savefig(hyperopt_plot_file, transparent=True)
    plt.show()


# In[31]:


def get_best_pipeline(hyperopt_output):
    '''
    Takes in 'best_model_choice' model/hyperparameter information from hyperopt fmin.
    Returns pipeline constructed per those optimized specifications.
    '''
    model_params = deepcopy(hyperopt_output)
    
    for key in hyperopt_output.keys():
        if type(model_params[key]) == dict:
            sub_dict = model_params.pop(key)
            model_params.update(sub_dict)
            
    print(model_params)
    
    model_type = model_options[model_params.pop('model_type')]
    
    if model_type == 'rfc':
        max_depth = int(model_params['rfc_max_depth'])
        n_estimators = int(model_params['rfc_n_ests'])
        class_weight = class_weight_options[int(model_params['rfc_class_wt'])]
        criterion = criterion_options[int(model_params['rfc_criterion'])]
        model = RandomForestClassifier(n_jobs=3,
                                       max_depth = max_depth,
                                       n_estimators = n_estimators,
                                       class_weight = class_weight,
                                       criterion = criterion
                                      )
    elif model_type == 'svc':
        C = model_params['svc_C']
        class_weight = class_weight_options[int(model_params['svc_class_wt'])]
        kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
        kernel = kernel_list[int(model_params['svc_kernel'])]
        gamma_name = 'svc_gamma_{}'.format(kernel)
        gamma = model_params[gamma_name]
        if kernel == 'poly':
            degree = model_params['svc_degree']
            model = SVC(probability=True,
                        C=C,
                        class_weight=class_weight,
                        kernel=kernel,
                        gamma=gamma,
                        degree=degree
                       )
        else:
            model = SVC(probability=True,
                        C=C,
                        class_weight=class_weight,
                        kernel=kernel,
                        gamma=gamma
                       )
        
    elif model_type == 'sgd':
        alpha = model_params['sgd_alpha']
        penalty = sgd_penalty_options[model_params['sgd_alpha']]
        l1_ratio = model_params['sgd_ratio']
        model = SGDClassifier(loss='log',
                              n_jobs=3,
                              alpha=alpha,
                              l1_ratio=l1_ratio
                             )
        
    elif model_type == 'gpc':
        length_scale = model_params['gpc_lscale']
        model = GaussianProcessClassifier(RBF(length_scale=length_scale))
        
    else:
        max_depth = int(model_params['dtc_max_depth'])
        criterion = criterion_options[model_params['dtc_criterion']]
        min_samples_leaf = model_params['dtc_min_samples']
        class_weight = class_weight_options[model_params['dtc_class_wt']]
        model = DecisionTreeClassifier(**model_params)
    
    pipe = make_pipeline(StandardScaler(), model)
    return pipe


# In[43]:


def perm_import(fitted_pipeline, X_test, y_test, class_dict):
    '''
    Takes in fitted model, test set, and dictionary of class labels and binary labels
    Returns log loss permutation importance for all features
    '''
    y_pred = fitted_pipeline.predict_proba(X_test)
    y_test_n = [class_dict[country] for country in y_test]
    y_test_n = pd.get_dummies(y_test_n)
    
    loss = log_loss(y_test_n, y_pred)
    #print(loss)
    
    feature_importances = {}
    for feat in X_test.columns:
        X_test_copy = deepcopy(X_test)
        feat_to_randomize = deepcopy(X_test[feat])
        feat_to_randomize = feat_to_randomize.sample(frac=1)
        feat_to_randomize.reset_index(inplace=True, drop=True)
        X_test_copy.loc[:,feat] = feat_to_randomize.values
        
        y_pred = fitted_pipeline.predict_proba(X_test_copy)
        feat_loss = log_loss(y_test_n, y_pred)
        feat_import = feat_loss - loss
        feature_importances[feat] = feat_import
    return feature_importances
    
    


# In[ ]:


def plot_confmx(y_test, y_pred, labels, confmx_filename):
    '''
    plots confusion matrix and saves it to file
    '''
    conf_mat = confusion_matrix(y_test, y_pred, labels)
    plt.figure(figsize = (8,5.5))
    conf_heatmap = sns.heatmap(conf_mat, annot=True, xticklabels=labels, 
                               yticklabels=labels,
                               cmap = sns.color_palette("GnBu_d"))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    conf_heatmap.figure.savefig(confmx_filename, transparent=True)


# In[ ]:


def rand_jitter(arr):
    stdev = .005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


# In[ ]:


def plot_data_by_feats(x_feat, y_feat, X, class_dict, feat_plot_filename):
    '''
    Takes in two features of interest, full dataset X, and a dictionary relating class labels to class id numbers
    Plots X data along specified feature dimensions
    '''
    x = X[x_feat]
    x = rand_jitter(x)
    y = X[y_feat]
    y = rand_jitter(y)
    label = pd.Series(X['Country.of.Origin']).apply(lambda x: class_dict[x])
    colors = ['#d98f5e', '#664C3e', '#6eb6be', '#9b98fd']
    countries = ['Brazil','Colombia','Guatemala','Mexico']

    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#fffbf0')

    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.5)
    #plt.xlim(6.5,8.5)
    plt.xlabel('Rating: '+ x_feat)
    plt.ylabel('Rating: '+ y_feat)

    c = [mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor=colors[i], edgecolor="none") for i in range(len(countries))]
    plt.legend(c,countries,loc=(1.01,0.5), facecolor='none', edgecolor='none')
    plt.savefig(feat_plot_filename, transparent=True)
    plt.show()
    


# In[ ]:


def model_performance_for_feature(feature, X_test, y_test, y_pred, class_dict, perf_by_feat_filename):
    '''
    Takes in one feature of interest, test data, predicted values, and dictionary relating class labels to class id numbers
    Plots test data, organized by true label, color-coded by predicted label, with selected feature values as x axis
    '''
    x = X_test[feature]
    x = rand_jitter(x)
    y = y_test.apply(lambda x: class_dict[x])
    y = rand_jitter(y)
    label = pd.Series(y_pred).apply(lambda x: class_dict[x])
    colors = ['#d98f5e', '#664C3e', '#6eb6be', '#9b98fd']
    countries = ['Brazil','Colombia','Guatemala','Mexico']

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#fffbf0')

    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.7)

    plt.yticks([0,1,2,3,4], countries)
    plt.ylabel('True Country of Origin')
    plt.xlabel('Rating: '+feature)

    cb = plt.colorbar()
    cb.set_label('Predicted', rotation=270)
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks([0.5,1,2,2.5])
    cb.set_ticklabels(countries)
    plt.savefig('label_scatter1.png', transparent=True)

