from parameters import save_boolean
import pandas as pd
from parameters import date, repository_path, folder, date, scoring
import matplotlib.pyplot as plt
import os, sys
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, auc, precision_recall_fscore_support, pairwise, f1_score, log_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
import numpy as np
from pathlib import Path


### save functions
def save_df(df, df_name='default', rel_path='/data/final/'):
    """
    simple function for saving result table. uses the date and supplied df name and saves to the savepath specified above.
    """
    global folder
    
    save_path= str(repository_path)+rel_path
    
    address=save_path+'{}_{}/'.format(date,folder)
    if not os.path.exists(address):
        print(address)
        os.makedirs(address)
    else:
        print(address)
        
    if address.endswith('/')==False:
        address= address+'/'
        
    if df_name == 'default':
        df_name =[x for x in globals() if globals()[x] is df][0]
        
    pd.DataFrame(df).to_csv(Path(address+'{}_{}.csv'.format(date, df_name)))

def saveplot(plt, figure_name):
    """
    simple function for saving plots
    """
    address = str(repository_path)+'/figures/{}_{}'.format(date,folder)
    print(address)

    if not os.path.exists(address):
        os.makedirs(address)
    plt.savefig(address+"/{}.png".format(figure_name),bbox_inches='tight')


### model evaluation & plotting functions

def evaluate(model, x, y):
    "simple classification evaluation metrics and output used in my hypertuning functions"
    from sklearn.metrics import log_loss
    
    y_hat = model.predict(x)
    y_hat_proba = model.predict_proba(x)[:, 1] 
    errors = abs(y_hat - y)
    mape = 100 * np.mean(errors / y)
    accuracy = 100 - mape
    auc=roc_auc_score(y, y_hat_proba)
    loss= log_loss(y, y_hat_proba)
        
    print ('the AUC is: {:0.3f}'.format(auc))
    print ('the logloss is: {:0.3f}'.format(loss))
    print(confusion_matrix(y, y_hat))
    print(classification_report(y,y_hat, digits=3))
    
    if scoring=='neg_log_loss':
        return_value=loss
    elif scoring=='roc_auc':
        return_value=auc
    else:
        raise ValueError
    
    return (return_value)

def optimal_youden_index(fpr, tpr, thresholds, tp90=True):
    """
    inputs fpr, tpr, thresholds from metrics.roc(),
    outputs the clasification threshold, roc dataframe, and the index of roc dataframe for optimal youden index
    """
    #making dataframe out of the thresholds
    roc_df= pd.DataFrame({"thresholds": thresholds,"fpr":fpr, "tpr": tpr})
    roc_df.iloc[0,0] =1
    roc_df['yuden']= roc_df['tpr']-roc_df['fpr']
    
    if tp90==True:
        idx= roc_df[roc_df['tpr']>=0.9]['yuden'].idxmax() #changed this so now finds optimial yuden threshold but tp>=90%
    else:
        idx=roc_df['yuden'].idxmax() #MAX INDEX
    
    youden_threshold=roc_df.iloc[idx,0] #threshold for max youden
    return(youden_threshold, roc_df, idx)
    
def plot_roc(fpr, tpr, roc_auc, roc_df, idx, save=save_boolean, model_name=None, folder_name=None, file_name=None):
    plt.title('ROC with optimal Youden Index')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    
    #finding the point on the line given threshold 0.5 (finding closest row in roc_df)
    og_idx=roc_df.iloc[(roc_df['thresholds']-0.5).abs().argsort()[:1]].index[0]
    plt.plot(roc_df.iloc[og_idx,1], roc_df.iloc[og_idx,2],marker='o', markersize=5, color="g")
    plt.annotate(s="P(>=0.5)",xy=(roc_df.iloc[og_idx,1]+0.02, roc_df.iloc[og_idx,2]-0.04),color='g') #textcoords
    
    plt.plot(roc_df.iloc[idx,1], roc_df.iloc[idx,2],marker='o', markersize=5, color="r") ##
    plt.annotate(s="TPR>=0.9",xy=(roc_df.iloc[idx,1]+0.02, roc_df.iloc[idx,2]-0.04),color='r' ) #textcoords
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.2)
    
    if save==True:
        saveplot(plt, figure_name="{}_roc".format(model_name))
    else: pass
    
    plt.show()
    
    
def classifier_eval(model, x, y, proba_input=False,pos_label=1, print_default=True,model_name=None, folder_name=None, save=save_boolean):
    import sklearn.metrics as metrics
    from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score
    """
    classification evaluation function. able to print/save the following:
    
    print/save the following:
        ROC curve marked with threshold for optimal youden (maximizing tpr+fpr with constraint that tpr>0.9)

        using 0.5 threshold:
            confusion matrix
            classification report
            npv
            accuracy

        using optimal youden (maximizing tpr+fpr with constraint that tpr>0.9):
            confusion matrix
            classification report
            npv
            accuracy
    
    output: 
        outputs modelname, auc, precision, recall, f1, and npv to a dictionary. 
    
    notes:
    youden's J statistic:  J= sensitivity + specificity -1
    (truepos/ truepos+falseneg) + (true neg/ trueneg + falsepos) -1. 
    NOTE: with tpr>0.9 turned on, the youden statistic is basically just the furthest point on the line away from the midline with tpr>=0.9
    NOTE2: this function arguably does too much. in the future it may be better to seperate it out into more compartmental functions like with preprocessing().
    """
    
    if proba_input==True: 
        y_proba= model
        y_pred=[1 if y >= 0.5 else 0 for y in y_proba]
    
    else:
        model_name=type(model).__name__

        y_pred = model.predict(x)
        y_proba = model.predict_proba(x)[:,1]
        
    fpr, tpr, thresholds = metrics.roc_curve(y, y_proba, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)
#     print("AUROC:",roc_auc)
    
    #gathering the optimal youden_index and df of tpr/fpr for auc and index of that optimal youden. idx is needed in the roc
    youden_threshold, roc_df, idx= optimal_youden_index(fpr, tpr, thresholds,tp90=True)

    #plotting roc
    plot_roc(fpr, tpr, roc_auc, roc_df, idx, save=save, model_name=model_name,folder_name=folder)
    plt.show(), plt.close()
    
    #printing npv, recall, precision, accuracy
    npv=confusion_matrix(y, y_pred)[0,0]/sum(np.array(y_pred)==0)
    prec= precision_score(y_true=y, y_pred= y_pred, pos_label=pos_label)
    recall= recall_score(y_true=y, y_pred= y_pred, pos_label=pos_label)
    f1= f1_score(y_true=y, y_pred= y_pred, pos_label=pos_label)
    
    if print_default==True: ###can opt to not print the 0.5 classification threshold classification report/conf matrix
        #plotting confusion matrixs
        print("******* Using 0.5 Classification Threshold *******\n")
        print(confusion_matrix(y, y_pred))
        print ('the Accuracy is: {:01.3f}'.format(accuracy_score(y, y_pred)))
        print ("npv: {:01.3f}".format(npv))
        print ('the classification_report:\n', classification_report(y,y_pred, digits=3))
    else:
        pass
    
    #### YOUDEN ADJUSTMENT #####

    print("******* Using Optimal Youden Classification Threshold *******")
    print("\nthe Youden optimal index is : {:01.3f}".format(youden_threshold))

    y_pred_youden = [1 if y >= youden_threshold else 0 for y in y_proba]

    npv_y=confusion_matrix(y, y_pred_youden)[0,0]/sum(np.array(y_pred_youden)==0)
    prec_y= precision_score(y_true=y, y_pred= y_pred_youden, pos_label=pos_label)
    recall_y= recall_score(y_true=y, y_pred= y_pred_youden, pos_label=pos_label)
    f1_y= f1_score(y_true=y, y_pred= y_pred_youden, pos_label=pos_label)
    auc_y=roc_auc_score(y_true=y, y_score= y_proba)
    
    ##plotting and saving confusion matrix
    confusion_youden=confusion_matrix(y, y_pred_youden)
    
    #plotting confusion matrixs
    print(confusion_matrix(y, y_pred_youden))
    print ('the Accuracy is: {:01.3f}'.format(accuracy_score(y, y_pred_youden)))
    print ("npv: {:01.3f}".format(npv_y))
    print ('the classification_report:\n', classification_report(y,y_pred_youden, digits=3))
    
    youden_dic= {'model':model_name, 'auc':auc_y, 'precision':prec_y, 'recall':recall_y, 'f1':f1_y, 'npv':npv_y}
    return(youden_dic)
    
    
##### stacked roc plots

def roc_publishing(model, x, y, proba_input=False,pos_label=1, print_default=True, model_name=None):
    import sklearn.metrics as metrics
    from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score

    model_name=type(model).__name__

    y_proba = model.predict_proba(x)[:,1]
        
    fpr, tpr, thresholds = metrics.roc_curve(y, y_proba, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)
    
    #gathering the optimal youden_index and df of tpr/fpr for auc and index of that optimal youden. idx is needed in the roc
    youden_threshold, roc_df, idx= optimal_youden_index(fpr, tpr, thresholds, tp90=True)
    
    return(fpr, tpr, roc_auc, roc_df, idx)
    

def stacked_roc(x_test, y_test, models_dic, first_bold=True, plot_threshold=False):
    """
    plotting function to plot a stacked ROC based on models in a dictionary. 
    first_bold=True means that the first model in the dic will stand out and be a solid line, while others are dotted
    """
    
    global save_boolean
    
    plt.style.use('seaborn-white')
    plt.rcParams['figure.figsize'] = [7, 4]
    
    if first_bold==True:
        i=0
    else: 
        i=1
        
    for model_name in models_dic.keys():
        if i==0:
            model=models_dic[model_name]
            fpr, tpr, roc_auc, roc_df, idx= roc_publishing(model, x=np.array(x_test), y=y_test, model_name=model_name)
            # print(model_name, roc_auc)
            ax1= plt.plot(fpr, tpr, 'b', label = '%s AUC = %0.3f' % (model_name, roc_auc), linewidth=2)
            og_idx=roc_df.iloc[(roc_df['thresholds']-0.5).abs().argsort()[:1]].index[0]
            if plot_threshold==True:
                plt.plot(roc_df.iloc[og_idx,1], roc_df.iloc[og_idx,2],marker='o', markersize=8, color="black")
                plt.plot(roc_df.iloc[idx,1], roc_df.iloc[idx,2],marker='o', markersize=6, color='r') ##

        else:
            model=models_dic[model_name]
            fpr, tpr, roc_auc, roc_df, idx= roc_publishing(model, x=np.array(x_test), y=y_test, model_name=model_name)
            # print(model_name, roc_auc)
            ax1= plt.plot(fpr, tpr, label = '%s AUC = %0.3f' % (model_name, roc_auc), linestyle='dotted')
            og_idx=roc_df.iloc[(roc_df['thresholds']-0.5).abs().argsort()[:1]].index[0]
            if plot_threshold==True:
                plt.plot(roc_df.iloc[og_idx,1], roc_df.iloc[og_idx,2],marker='o', markersize=8, color="black")
                plt.plot(roc_df.iloc[idx,1], roc_df.iloc[idx,2],marker='o', markersize=6, color='r') ##
    i+=1
        
    ###annotating the plot
    plt.legend(loc = 'lower right')   
    if plot_threshold==True:
        plt.annotate(s="P(0.5)",xy=(0.71, 0.50),color='black', size=10) #textcoords #alt: xy=(0.78, 0.345)
        plt.plot(0.68, 0.51, 'ro', color='black') #alt: (0.73, 0.36, 'ro', color='black')
        plt.annotate(s="P(tuned)",xy=(0.71, 0.56),color='black', size=10) #textcoords #alt: xy=(0.78, 0.405)
        plt.plot(0.68, 0.57, 'ro', color='r') #alt: (0.73, 0.42, 'ro', color='r')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', size=14)
    plt.xlabel('False Positive Rate', size=14)

    if save_boolean==True:
        saveplot(plt,'stacked_roc')
    else: pass
    plt.show()
    
    
##### variable importance fxns


def find_N_varimp_set(x_train, models_dic):
    """
    function that takes in a dictionary of models and the x_train dataframe and returns the set of variables present in the combined list of each model's top N most important variables.
    1) find top N variables for each model
    2) make list of all models top N
    3) filter to only unique values in list = varimp_names
    """
    global n_varimp
    features_dic={}
    top_set_dic={}

    for model_name in models_dic.keys():
        model= models_dic[model_name]
        print(model_name)
        if model_name in ['knn','ensemble', 'mlp']:
            pass
        elif model_name in ['logreg','svc']:
            feature_importance = abs(model.coef_[0])
            sorted_idx = np.argsort(feature_importance)[-n_varimp:]#[0]
            features =list(np.array(x_train.columns)[sorted_idx][-n_varimp:])
            features_dic.update( {model_name :features } )
        else:
            feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
            features=feat_importances.nlargest(n_varimp).sort_values()
            features=list(features.reset_index()['index'])
            features_dic.update( {model_name :features } )
    #######
    set_features=[]

    for features in features_dic.values():
        set_features=set_features+features
    set_features=set(set_features)
    varimp_names=list(set_features)

    return(varimp_names)
    
def topN_rel_imp(models_dic, varimp_names):
    """
    input:dictionary of models and the top N set of important variables among models
    output: relative variable importance for each model of all set(varimp_names) variables.
    note: relative variable importance determined by dividing each variable importance by the value of the most important variable. this makes all values a comparison to the most important varaible:
    ie 50 rel variable importance = half as important as the most important variable
    """
    
    # finding the index of the set(varimp_names) in the dataframe.  
    #getting index of the set(top10) variables in x_train
    xtrain_column_index_list=[]
    for element in varimp_names:
        variable_index=list(x_train).index(element)
        xtrain_column_index_list.append(variable_index)
    
    top_set_dic={} #instantiating dictionary
    for model_name in models_dic.keys(): ##now that we have set of top N variables for each model. we can make relative importance for all unique variables in the set
        model= models_dic[model_name]
        if model_name in ['knn','ensemble', 'mlp']:
            pass
        
        elif model_name in ['logreg','svc']: 
            imp= abs(model.coef_[0])[xtrain_column_index_list]
            rel_imp=100.0 * (imp / imp.max())
            features =list(np.array(x_train.columns)[xtrain_column_index_list])#[-n_varimp:])
            top_set= pd.Series(rel_imp,features).sort_values()
            top_set_dic.update( {model_name :top_set } )

        else:
            imp=pd.Series(models_dic[model_name].feature_importances_, index=x_train.columns)[xtrain_column_index_list]
            imp=imp.sort_values()
            rel_imp=100.0 * (imp / imp.max())
            features =list(np.array(x_train.columns)[xtrain_column_index_list])
            top_set= rel_imp
            top_set_dic.update( {model_name :top_set } )

    return(top_set_dic)
    

def roc_name_adjust(varimp_names):
    """
    cleans up the column names for the variable importance plot for publishing
    """
    adjusted_names=[]
    mapper={'vent_recieved_2.0': 'mechanical ventilation recieved',
            'vent_recieved_1.0': 'oxygen ventilation recieved',
            'vent_recieved_1.0': 'no ventilation recieved',
            'pao2fio2ratio':'PaO2:FiO2',
            'ipco2_>50': 'pCO2 (>50)',
            'ibands_>10': 'bands (>10)',
            'ibands_absent': 'bands (missing)'}
    
    for element in varimp_names:
        if element in mapper.keys():
            element= mapper[element]
            adjusted_names.append(element)
        elif "_1.0" in element:
            element= element.strip("_1.0") + ' (Y/N)'
            adjusted_names.append(element)
        else:
            adjusted_names.append(element)
        
    return(adjusted_names)
    

def plot_topN_rel_imp(top_set_dic, varimp_names, xvar_rotation=80):
    """
    plot the variable importance plots as a lineplot
    rotation: the amount of xvar rotation
    """
    global save_boolean
    
    df_base=pd.DataFrame(index=varimp_names)  

    for model_name in top_set_dic.keys():
        df_base[model_name]= top_set_dic[model_name]

    adjusted_names=roc_name_adjust(varimp_names)
    df_base.index=adjusted_names
    df_base.sort_values('rf', ascending=False)

    plt.style.use('seaborn-ticks')
    plt.rcParams['figure.figsize'] = [10,10]#[7, 7]
    plt.plot(df_base.sort_values('rf', ascending=True))
    #plt.set_xticklabels(adjusted_names,rotation=30)
    plt.xticks(rotation=xvar_rotation)#, ha='right')
    plt.ylabel("Relative Variable Importance")
    plt.legend(list(df_base))
    
    if save_boolean==True:
        saveplot(plt,'variable_importance')

    return(df_base)


##### hypertuning functions
def hypertuning_fxn(x, y, nfolds, model , param_grid, subject_id, scoring=scoring, gridsearch=True, n_iter=20, verbose=False): 
    """
    use this function to hypertune parameters of machine learning models given a parameter grid.
    gridsearch=True means it will test all combinations
    gridsearch=False; n_iter: will test n_iter pseudo-random hyperparameter combinations

    in general the function will present the average scorer "default=AUROC" values across all cv folds for the best and worst parameters.
    Additionally it will also take the best and worst parameters and will fit & predict the trainingset. this is useful to see overfitting.
    """
    from sklearn.model_selection import GroupKFold

    print("######## model: ", type(model).__name__, '########')

    np.random.seed(12345)
    if gridsearch==True:
        grid_search = GridSearchCV(estimator= model,
                                         param_grid=param_grid,
                                         cv=GroupKFold(nfolds),
                                         scoring=scoring,
                                         return_train_score=True,
                                         n_jobs = -1)
    else:
        grid_search = RandomizedSearchCV(estimator= model,
                                         param_distributions= param_grid,
                                         n_iter=n_iter,
                                         cv=GroupKFold(nfolds),
                                         scoring=scoring,
                                         return_train_score=True,
                                         random_state=12345,
                                         n_jobs = -1)
    
    # if type(model).__name__=='XGBClassifier':
    #     grid_search.fit(np.array(x), np.array(y).ravel(), groups=subject_id)    
    # else:
    #     grid_search.fit(x, y, groups=subject_id)    
    grid_search.fit(x, y, groups=subject_id)    
    print(" scorer function: {}".format(scoring))
    print(" ##### CV performance: mean & sd scores #####")

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    print('best cv score: {:0.3f}'.format(grid_search.best_score_))
    print('best cv params: ', grid_search.best_params_)

    worst_index=np.argmin(grid_search.cv_results_['mean_test_score'])
    print('worst cv score: {:0.3f}'.format(grid_search.cv_results_['mean_test_score'][worst_index]))
    print('worst cv params: ', grid_search.cv_results_['params'][worst_index])
    ##
    if verbose==True:
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

    print('##### training set performance #####\n')   
    print(' best hypertuned model training set performance:')
    best_random = grid_search.best_estimator_
    best_random_auc = evaluate(best_random, x, y)
    
    print(' worst hypertuned model training set performance:')
    worst_params= grid_search.cv_results_['params'][worst_index]
    worst_random=model.set_params(**worst_params)
    worst_random.fit(x,y)
    worst_random_auc = evaluate(worst_random, x, y)      
          
    print('relative scorer change of {:0.2f}%. between worst and best hyperparams on TRAINING set (may be overfit)'.format( 100 * (best_random_auc - worst_random_auc) / worst_random_auc))
    
    return(grid_search)

def hypertuned_cv_fxn(x, y, model_in, nfolds, subject_id):
    """
    the goal of this function is to take the best hypertuned model and 
    generate average and std for F-1, precision, recall, npv, and AUC across each fold.
    Ideally i could have generated this above in my hypertuning cv function,
    but it actually took less computational time to just rerun cv on the best performing evaluator and collect all of the averaged performance metrics
    """
    from sklearn.model_selection import GroupKFold
    import sklearn.metrics as metrics
    from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score
    from sklearn.base import clone

    pos_label=1
    model= clone(model_in, safe=True)
    np.random.seed(12345)
    group_kfold = GroupKFold(n_splits=nfolds)
    group_kfold.get_n_splits(x, y, subject_id)

    f1_y_cv=[]
    auc_y_cv=[]
    prec_y_cv=[]
    recall_y_cv=[]
    npv_y_cv=[]

    for train_index, test_index in group_kfold.split(x, y, subject_id):
        x_train_cv, x_test_cv = x[train_index], x[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        model.fit(x_train_cv, y_train_cv)
        
        y_proba = model.predict_proba(x_test_cv)[:,1]
        y_pred = model.predict(x_test_cv)

        fpr, tpr, thresholds = metrics.roc_curve(y_test_cv, y_proba, pos_label=pos_label)    
        #gathering the optimal youden_index and df of tpr/fpr for auc and index of that optimal youden. idx is needed in the roc
        youden_threshold, roc_df, idx= optimal_youden_index(fpr, tpr, thresholds,tp90=True)
        y_pred_youden = [1 if y >= youden_threshold else 0 for y in y_proba]
            
        npv_y=confusion_matrix(y_test_cv, y_pred_youden)[0,0]/sum(np.array(y_pred_youden)==0)
        npv_y_cv.append(npv_y)

        prec_y= precision_score(y_true=y_test_cv, y_pred= y_pred_youden, pos_label=pos_label)
        prec_y_cv.append(prec_y)

        recall_y= recall_score(y_true=y_test_cv, y_pred= y_pred_youden, pos_label=pos_label)
        recall_y_cv.append(recall_y)

        f1_y= f1_score(y_true=y_test_cv, y_pred= y_pred_youden, pos_label=pos_label)
        f1_y_cv.append(f1_y)

        ###need to debug this.###
        auc_y=roc_auc_score(y_true=y_test_cv, y_score= y_proba)
        auc_y_cv.append(auc_y)
        
        youden_dic_cv= {'model':type(model).__name__, 
                'auc':np.mean(auc_y_cv),
                'auc_sd':np.std(auc_y_cv),
                'precision':np.mean(prec_y_cv),
                'precision_sd':np.std(prec_y_cv),
                'recall':np.mean(recall_y_cv),
                'recall_sd':np.std(recall_y_cv),
                'f1':np.mean(f1_y_cv),
                'f1_sd':np.std(f1_y_cv),
                'npv':np.mean(npv_y_cv),
                'npv_sd':np.std(npv_y_cv)}
        
    return(youden_dic_cv)

