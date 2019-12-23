import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
import seaborn as sns
import numpy as np
import glob
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, auc, precision_recall_fscore_support, pairwise, f1_score, log_loss
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Memory
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, Imputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils import validation
from scipy.sparse import issparse
from scipy.spatial import distance
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier #conda install -c conda-forge xgboost to install

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore') #ignore all warnings

RANDOM_STATE = 15485867
plt.style.use('seaborn-white')

from modeling_fxn import saveplot, optimal_youden_index, plot_roc, save_df
from modeling_fxn import hypertuning_fxn, hypertuned_cv_fxn
from parameters import nfolds, scoring, n_iter, gridsearch
#patients of interest from rotation_cohort_generation
from parameters import final_pt_df_v, date, repository_path, lower_window, upper_window, folder, date, time_col, time_var, patient_df

#from parameters import save_boolean
save_boolean=False


final_pt_df2 = final_pt_df_v #pd.read_csv('/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/raw/csv/%s_final_pt_df2.csv'%(most_updated_patient_df), index_col=0)
del(final_pt_df_v)
patients= list(final_pt_df2['subject_id'].unique())
hadm_id= list(final_pt_df2['hadm_id'].unique())
icustay_id= list(final_pt_df2['icustay_id'].unique())
icustay_id= [int(x) for x in icustay_id]


####fxns#####

def data_import(allFiles):
    """
    function to import x_train, x_test, y_train, and y_test using glob of the data/final folder.
    """
    for name in allFiles:
        if 'test' in name:
            if 'x_' in name:
                x_test = pd.read_csv(name,  index_col=0)
            elif 'icustay_' in name:
                test_icustay_id= pd.read_csv(name,  index_col=0)
            else:
                 y_test = pd.read_csv(name,  index_col=0)

        elif 'train' in name:
            if 'x_' in name:
                x_train = pd.read_csv(name,  index_col=0)
            elif 'icustay_' in name:
                train_icustay_id= pd.read_csv(name,  index_col=0)
            else:
                 y_train = pd.read_csv(name,  index_col=0)
    return(x_train, x_test, y_train, y_test, train_icustay_id, test_icustay_id)


def xgboost_h(x,y, subject_id):
    """ function to hypertune xgboost"""
    model= XGBClassifier(n_estimators=100, min_child_weight=2, #changed: GridSearchCV ->RandomizedSearchCV
                                              gamma=0, subsample=0.8, colsample_bytree=0.8,
                                              objective='binary:logistic', n_jobs=-1, seed=27)
    scale_pos_weight = [1, 5, 10] #0.1
    max_depth = [1, 2, 3, 4, 5]
    learning_rate=[0.01, 0.1, 0.5, 1]
    param_grid = {'scale_pos_weight': scale_pos_weight, 'max_depth' : max_depth, "learning_rate":learning_rate}
    #np.array(y).ravel()
    xgboost_hyper=hypertuning_fxn(x, y, nfolds=nfolds, model=model , param_grid=param_grid, subject_id=subject_id, scoring=scoring, n_iter=n_iter, gridsearch=gridsearch)
    xgboost_cv= hypertuned_cv_fxn(x, y, xgboost_hyper.best_estimator_, nfolds=nfolds, subject_id=subject_id)
    return(xgboost_hyper, xgboost_cv)

def rf_h(x,y, subject_id):
    ###rf
    #{'bootstrap': False, 'class_weight': None, 'max_depth': 25, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200} 
    # Number of trees in random forest
    n_estimators = [10,15, 25, 50, 150, 250] #[int(x) for x in np.linspace(start = 10, stop = 1000, num = 5)]
    # Number of features to consider at every split
    max_features = [3,10,20,'auto']
    # Maximum number of levels in tree
    max_depth = [5,10, 25]#[int(x) for x in np.linspace(5, 110, num = 5)]
    #max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 5, 10]
    # Method of selecting samples for training each tree. supposedly better with false when classes aren't perfectly ballanced
    bootstrap = [True, False] #[True, False] #[False] 
    #### note about bootstrap= off
    ###Without bootstrapping, all of the data is used to fit the model, so there is not random variation between trees with respect to the selected examples at each stage. 
    ###However, random forest has a second source of variation, which is the random subset of features to try at each split.
    ### The documentation states "The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)," 
    ### which implies that bootstrap=False draws a sample of size equal to the number of training examples without replacement, i.e. the same training set is always used.
    ### we test this turned off because with unballanced classes turning this off can sometimes improve performance

    #class_weight is either a dictionary of each class to a uniform weight for that class (e.g., {1:.9, 2:.5, 3:.01}), or is a string telling sklearn how to automatically determine this dictionary.
    class_weight= [None, {0:(1/np.bincount(y))[0], 1:(1/np.bincount(y))[1]}]

    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'class_weight': class_weight,
                   'bootstrap': bootstrap}

    model= RandomForestClassifier(criterion='entropy', random_state=12345)

    #rf_hyper=hypertuning_fxn(x, y, nfolds=nfolds, model=model , param_grid=param_grid, scoring=scoring,n_iter = n_iter, gridsearch=False)
    rf_hyper=hypertuning_fxn(x, y, nfolds=nfolds, model=model , param_grid=param_grid,subject_id=subject_id, scoring=scoring,n_iter = n_iter, gridsearch=False)
    rf_cv= hypertuned_cv_fxn(x, y, rf_hyper.best_estimator_, nfolds=nfolds, subject_id=subject_id)
    return(rf_hyper, rf_cv)

def logreg_h(x,y, subject_id):
    ###logreg
    model= LogisticRegression(solver='lbfgs',random_state=12345)
    #model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=None)

    penalty = ['l1','l2']
    class_weight=['balanced',None]

    param_grid = {'penalty': penalty,
                  'class_weight': class_weight}

    logreg_hyper=hypertuning_fxn(x, y, nfolds=nfolds, model=model , param_grid=param_grid, subject_id=subject_id, scoring=scoring, n_iter=n_iter, gridsearch=True)
    logreg_cv= hypertuned_cv_fxn(x, y, logreg_hyper.best_estimator_, nfolds=10, subject_id=subject_id)
    return(logreg_hyper, logreg_cv)


def knn_h(x,y, subject_id):
    from sklearn.neighbors import KNeighborsClassifier
    model= KNeighborsClassifier()

    n_neighbors = [3,4,5, 8, 10, 25]
    weights=['uniform']
    p=[1,2] #1= mmanhattan, 2= euclidian

    param_grid = {'n_neighbors': n_neighbors,
                  'weights': weights,
                  'p': p}

    knn_hyper=hypertuning_fxn(x, y, nfolds=nfolds, model=model ,subject_id=subject_id, param_grid=param_grid, scoring=scoring, n_iter=n_iter, gridsearch=True)
    knn_cv= hypertuned_cv_fxn(x, y, knn_hyper.best_estimator_, nfolds=10, subject_id=subject_id)
    return(knn_hyper, knn_cv)

def mlp_h(x,y, subject_id):
    from sklearn.neural_network import MLPClassifier
    #hidden_layer_sizes = [(50,), (100,), (150,), (250,)] #origional used parameters #(50,50), (100,100), (150,150),(300,)
    hidden_layer_sizes =[(50), (50,50), (50,50,50), (50,100,50), (100,),(150)]
    solver= ['sgd', 'adam']
    #     'alpha': [0.0001, 0.05],
    #     'learning_rate': ['constant','adaptive'],

    activation= ['relu','tanh']#, 'logistic']

    alpha= [0.001, 0.05] #L2 penalization degree #[0.001, 0.01]

    learning_rate= ['adaptive']
    learning_rate_init= [0.001, 0.01]

    param_grid = {'hidden_layer_sizes': hidden_layer_sizes, 
                  'solver':solver,
                  'activation':activation,
                  'alpha': alpha,
                  'learning_rate': learning_rate,
                  'learning_rate_init': learning_rate_init
                 }

    model= MLPClassifier(early_stopping=True, random_state=12345)

    #removed the standard scaler transformation to work better with ensemble
    mlp_hyper=hypertuning_fxn(x, y, nfolds=nfolds, model=model ,subject_id=subject_id, param_grid=param_grid, scoring=scoring,n_iter = 80, gridsearch=True) #gridsearch=False: testing with smaller, randomized grid
    #gridsearch=False: testing with smaller, randomized grid. went from ~30 sec to 13min when ==True
    # adding in solver: sgd and adam 1.5min ->8min
    mlp_cv= hypertuned_cv_fxn(x, y, mlp_hyper.best_estimator_, nfolds=10, subject_id=subject_id)
    return(mlp_hyper, mlp_cv)


def ensemble_h(x,y, subject_id):
    global xgboost, logreg, rf, svc, knn, mlp

    from sklearn.ensemble import VotingClassifier
    #create a dictionary of our models
    estimators=[("xgboost", xgboost), ('rf', rf), ('log_reg', logreg), ('mlp',mlp), ('svc',svc)]
    #create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='soft', n_jobs=-1)
    ensemble.fit(x,y)

    #running the ensemble in cv
    ensemble_cv= hypertuned_cv_fxn(x, y, ensemble, nfolds=nfolds, subject_id=subject_id)
    return(ensemble, ensemble_cv)


def reset_model(model_name, hardcode=False):
    """
    MODEL FITTING. function to reset model to the best hyperparameter combination. 
    """
    global xgboost_hyper, logreg_hyper, rf_hyper, knn_hyper, mlp_hyper, svc_hyper
    
    if model_name== 'xgboost':
        model = xgboost_hyper.best_estimator_

    elif model_name== 'logreg':
        model = logreg_hyper.best_estimator_

    elif model_name== 'rf': 
        model = rf_hyper.best_estimator_

    elif model_name== 'svc':
        model = svc_hyper.best_estimator_

    elif model_name== 'knn':
        model = knn_hyper.best_estimator_
        
    elif model_name== 'mlp':
        model = mlp_hyper.best_estimator_
    return(model)

#### main #### 

def main():
    #importing x and y train and test
    print("type the hour window: 24, 48, or 72 (integer only)")
    date_window= "{}_hr_window".format(str(input()))

    allFiles = glob.glob(str(repository_path)+ '/data/final/{}_{}/'.format(date,date_window) + "*.csv")
    x_train, x_test, y_train, y_test, train_icustay_id, test_icustay_id= data_import(allFiles)

    train_subject_id= pd.merge(pd.DataFrame(train_icustay_id), final_pt_df2[['icustay_id','subject_id']], how='left')['subject_id'] #7205
    test_subject_id= pd.merge(pd.DataFrame(test_icustay_id), final_pt_df2[['icustay_id','subject_id']], how='left')['subject_id'] #7205

    x=np.array(x_train.copy())
    y=y_train.copy() #copy of y_train
    y=y.astype('int')
    y=np.array(y).reshape(1,-1).ravel() #converting y into a [ y, y, y, ... ] array

    ### hypertuning origional models
    xgboost_hyper, xgboost_cv= xgboost_h(x,y,train_subject_id)
    rf_hyper, rfcv= rf_h(x,y,train_subject_id)
    logreg_hyper, logreg_cv= logreg_h(x, y, train_subject_id)
    mlp_hyper, mlp_cv= mlp_h(x,y,train_subject_id)
    svc_hyper, svc_cv= svc_h(x,y,train_subject_id)
    knn_hyper, knn_cv= knn_h(x,y,train_subject_id)

    ### fitting the best performing hypertuned models
    xgboost = reset_model('xgboost')
    xgboost.fit(x, y)

    logreg = reset_model('logreg')
    logreg.fit(x, y)

    rf= reset_model('rf')
    rf.fit(x,y)

    svc= reset_model('svc')
    svc.fit(x, y)

    knn= reset_model('knn')
    knn.fit(x,y)

    mlp= reset_model('mlp')
    mlp.fit(x,y)

    ### ensembling the best tuned models
    ensemble, ensemble_cv= ensemble_h(x,y,train_subject_id)

    ### summarizing cv results
    cv_summary_df= pd.DataFrame([rf_cv,
                             logreg_cv,
                             xgboost_cv,
                             svc_cv,
                             knn_cv,
                             mlp_cv,
                             ensemble_cv])
    cv_summary_df= cv_summary_df.set_index('model').round(decimals=3).sort_values('auc', ascending=False)

    print('cv summary table:')
    print(cv_summary_df)

    if save_boolean==True:
        save_df(cv_summary_df, df_name='cv_summary_df', rel_path='/tables/')

    #saving models
    if save_boolean==True:
        model_save(xgboost,'xgboost')
        model_save(rf,'rf')
        model_save(logreg,'logreg')
        model_save(svc,'svc')
        model_save(knn,'knn')
        model_save(mlp,'mlp')
        model_save(ensemble,'ensemble')
    return()

main()
