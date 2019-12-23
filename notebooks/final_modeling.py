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

from modeling_fxn import saveplot, optimal_youden_index, plot_roc, classifier_eval, save_df
from modeling_fxn import stacked_roc, find_N_varimp_set, topN_rel_imp, roc_name_adjust, plot_topN_rel_imp #variable importance fxns
#patients of interest from rotation_cohort_generation
from parameters import final_pt_df_v, date, repository_path, lower_window, upper_window, folder, date, time_col, time_var, patient_df
from parameters import save_boolean
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
            else:
                 y_test = pd.read_csv(name,  index_col=0)
        elif 'train' in name:
            if 'x_' in name:
                x_train = pd.read_csv(name,  index_col=0)
            else:
                 y_train = pd.read_csv(name,  index_col=0)
    return(x_train, x_test, y_train, y_test)


def load_model(filename, timewindow):
    import pickle
    loaded_modle= pickle.load(open(filename, 'rb'))
    return(loaded_modle)




#### main #### 

def main():
    #importing x and y train and test
    # print("type the hour window: 24, 48, or 72 (integer only)")
    # input_str=input()
    # date_window= "{}_hr_window".format(str(input_str))

    allFiles = glob.glob(str(repository_path)+ '/data/final/{}_{}/'.format(date,folder) + "*.csv")
    x_train, x_test, y_train, y_test= data_import(allFiles)

    models = glob.glob(str(repository_path)+ '/models/{}_{}/'.format(date, folder)+'*')

    models_dic={}

    for model in models:
    	models_dic.update( {model.strip('.sav').split('_')[-1] : load_model(model, folder)} )
    	

    ### evaluating models
    df_list=[]
    #test_summary_df= pd.DataFrame(columns=[auc,f1,npv, precision, recall])
    print(models_dic.keys())
    for key in models_dic.keys():
        df_list.append(classifier_eval(models_dic[key], x=np.array(x_test), y=y_test, save=True))
    
    test_summary_df= pd.DataFrame(df_list).set_index('model').round(decimals=3).sort_values('auc', ascending=False)

    if save_boolean==True:
        save_df(test_summary_df, df_name='test_summary_df', rel_path='/tables/')
    print(test_summary_df)

    ### stacked roc
    stacked_roc(x_test, y_test, models_dic, first_bold=False, plot_threshold=False)


main()