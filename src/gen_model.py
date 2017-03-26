import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import itertools
import datetime
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.ensemble as ens
import os


def _make_dummy(df, cols):
    df_ = df.copy()
    for col in cols:
        for value in df[col].value_counts().nlargest(50).index:
            df_['{}_{}'.format(col, value)] = df[col] == value
        df_.drop(col, axis=1, inplace=True)
    return df_

def load_and_clean(csv_name):
    '''
    loads and cleans the dataset stored in the data folder of this repo
    '''
    loans = pd.read_csv('../data/{}'.format(csv_name))
    dropcols = ['id', 'member_id', 'funded_amnt_inv', 'grade', 'sub_grade', 'verification_status',
           'url', 'desc', 'title' , 'addr_state', 'dti', 'earliest_cr_line', 'inq_last_6mths',
            'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal'
           'revol_bal', 'revol_util', 'total_acc', 'total_pymnt', 'total_pymnt_inv', 'total_pymnt',
           'total_pymnt_inv','total_rec_prncp', 'total_rec_int','total_rec_late_fee', 'recoveries',
            'collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','next_pymnt_d',
            'last_credit_pull_d','collections_12_mths_ex_med','mths_since_last_major_derog','policy_code','annual_inc_joint'
            ,'dti_joint','verification_status_joint','acc_now_delinq','tot_coll_amt','tot_cur_bal'
            ,'open_acc_6m','open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il'
            ,'il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim'
            ,'inq_fi','total_cu_tl','inq_last_12m'
                   ]
    dropcols = set(dropcols)
    keep_cols = []
    for col in loans.columns:
        if col not in dropcols:
            keep_cols.append(col)
    y_col = 'delinq_2yrs'
    # 'delinq_2yrs' The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
    purpose_keeprows = ['small_business', 'credit_card', 'other']
    loans_ = loans[keep_cols]
    loans_clean = loans_[loans_['purpose'] == 'credit_card']
    loans_clean.dropna(inplace=True)
    loans_y = loans_clean.pop(y_col)
    loans_clean_ = loans_clean.copy()
    loans_clean_['issue_d'] = loans_clean['issue_d'].apply(pd.Timestamp)
    loans_clean_['year'] = loans_clean_['issue_d'].apply(lambda x: x.year)
    loans_clean_['zip_code'] = loans_clean_['zip_code'].astype('category')
    loans_clean_.drop(['application_type', 'purpose', 'initial_list_status', 'revol_bal',
                    'out_prncp_inv', 'revol_bal', 'pymnt_plan', 'issue_d', 'loan_status'], axis=1, inplace=True)
    rows_need_dummies = ['emp_title', 'home_ownership', 'emp_length', 'term', 'zip_code']
    loans_model = _make_dummy(loans_clean_, rows_need_dummies)
    return loans_model, loans_y

def generate_roc_curve(labels, probs, n_iter=500, curve_name=False):
    '''
    mdata must contain matched yte values
    --------
    returns
    rocs: list of dicts contining 
    '''
    if not curve_name:
        name = os.listdir('../images').pop
        name = name.replace('ROC-', '')
        name  = name + '-1'
        curve_name  = np.random.choice(np.range(500))
    thres_list = np.linspace(0,1,n_iter)
    thres_list = (thres_list * thres_list)
    rocs = []
    test = labels.reshape(-1,1)
    proba = probs.reshape(-1,1)
    pframe = pd.DataFrame(np.append(proba, test, axis=1), columns= ['predict', 'label'])
    c_true = pframe[pframe['label'] == 1]
    c_false = pframe[pframe['label'] != 1]
    for thres in thres_list:
        tpr = float((c_true['predict'] > thres).sum()) / c_true.shape[0]
        fpr = float((c_false['predict'] > thres).sum()) / c_false.shape[0]
        rocs.append({'tpr' : tpr, 'fpr': fpr})
    fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    fpr = []
    tpr = []
    for row in rocs:
        fpr.append(row['fpr'])
        tpr.append(row['tpr'])
    ax.scatter(fpr, tpr, color=colors.next());
    ax.plot(np.linspace(0,1,10), np.linspace(0,1,10));
    ax.set_xlim(0,1);
    ax.set_ylim(0,1);
    ax.legend
    fig.savefig('images/ROC-{}.png'.format(curve_name))
    return rocs

colors = itertools.cycle(['green', 'red', 'cyan', 'orange', 'violet',
                          'yellow', 'blue', 'pink'])


def roc_curve_many(json_folder, save_file=True):
    '''
    does the same as roc curve but for many items
    '''
    if json_folder[-1] != '/':
        json_folder += '/'
    colors  = itertools.cycle(['red','cyan', 'green', 'orange', 'violet', 'blue', 'pink'])
    if save_file:
        save_file = '../images/roc_curve-{}'.format(datetime.datetime.now().toordinal())
    list_of_pd_frames = os.listdir(json_folder)
    fig, ax = plt.subplots(1,1,figsize=(13,13))
    for frame in list_of_pd_frames:
        model = pd.read_json(json_folder+frame)
        model['delinq_2yrs'] = model['delinq_2yrs'].fillna(0)
        model['delinq_2yrs'] = model['delinq_2yrs'].apply(lambda x: 1 if x > 0 else 0)
        model.columns = ['probability', 'label']
        roc_values = []
        for thres in np.linspace(.01, .99, 500):
            model['pred'] = model['probability'].apply(lambda x: 1 if x > thres else 0)
            rowdict = {}
            is_pos = model[model['label'] == 1]
            is_neg = model[model['label'] == 0]
            rowdict['tpr'] = float(is_pos[is_pos['pred'] == 1].shape[0])/is_pos.shape[0]
            rowdict['fpr'] = float(is_neg[is_neg['pred'] == 1].shape[0])/is_neg.shape[0]
            roc_values.append(rowdict)
        roc_values = pd.DataFrame(roc_values)
        ax.scatter(roc_values['fpr'], roc_values['tpr'], color =colors.next(),label=frame)
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), color='black')
    ax.legend()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.savefig(save_file)

# loans, y = load_and_clean('loan.csv')

# model = ens.GradientBoostingClassifier(learning_rate=.1,n_estimators=50, max_depth=2)
# Xtr,Xte, ytr, yte = ms.train_test_split(loans, y)
# model.fit(Xtr, ytr)

# probs = model.predict_proba(Xte)
# probs = probs[:,1]

# generate_roc_curve(yte, probs, curve_name='GBR500')

# for num in np.arange(1,205,40):
#     model = ens.RandomForestClassifier(n_estimators=num, max_depth=20,
#                                                   n_jobs=-1)
#     Xtr,Xte, ytr, yte = ms.train_test_split(loans, y)
#     model.fit(Xtr, ytr)
#     probs = model.predict_proba(Xte)
#     probs = probs[:,1]
#     probs = pd.Series(probs)
#     labels= pd.Series(yte)
#     pd.concat([probs,labels], axis=1).to_json('../data/{}_tree_rf'.format(num))

# for num in np.arange(1,205,40):
#     model = ens.GradientBoostingClassifier(learning_rate=.1, n_estimators=num, max_depth=2)
#     Xtr,Xte, ytr, yte = ms.train_test_split(loans, y)
#     model.fit(Xtr, ytr)
#     probs = model.predict_proba(Xte)
#     probs = probs[:,1]
#     probs = pd.Series(probs)
#     labels= pd.Series(yte)
#     pd.concat([probs,labels], axis=1).to_json('../data/{}_tree_GBT'.format(num))