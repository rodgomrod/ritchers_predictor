import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import gc

INPUT = '../data/'

X_train = pd.read_csv(f'{INPUT}train_values.csv')

y_train = pd.read_csv(f'{INPUT}train_labels.csv').damage_grade

train_cols = X_train.columns
cat_list = list()

for col in train_cols:
    if X_train[col].dtypes == 'object':
        cat_list.append(col)
        

for col in cat_list:
    X_train[col] = (X_train.groupby(col).size())/X_train.shape[0]
    
    
#def custom_loss(y_pred, y_true):
#    y_pred = [list(x).index(max(x))+1 for x in np.array(y_pred).reshape((int(len(y_pred)/3), 3))]
#    F1_Score = f1_score(y_true, y_pred, average='micro')
##    if F1_Score != F1_Score:
##        F1_Score = 0
#    return 'F1_Score', F1_Score, True

def evaluate_microF1_lgb(truth, predictions):
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='micro')
    return ('macroF1', f1, True) 


# FEAUTRES IMPORTANCES
#X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train,
#                                                        test_size = 0.33,
#                                                        random_state = 42)

rfc = RandomForestClassifier()
rfc.fit(X_train.fillna(-1), y_train)
features = pd.Series(rfc.feature_importances_, index= X_train.columns)
features = features.sort_values(ascending=False)

ft_sel = list(features[:15].index)
ft_sel

X_train2 = X_train[ft_sel]
X_train2 = X_train

    
k = 3
train_ids = X_train2.index
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
skf.get_n_splits(train_ids, y_train)

ft_importances = np.zeros(X_train2.shape[1])
b_r = 0

params = {
        "objective" : "multiclass",
          "num_class" : 3,
          "num_leaves" : 50,
          "max_depth": 7,
          "learning_rate" : 0.05,
          "bagging_fraction" : 0.95,  # subsample
          "feature_fraction" : 0.95,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 42,
          "random_state" : 42,
          "num_boost_round": 10000,
          "n_jobs": -1,
          "min_child_samples": 10,
          "min_child_weight": 0.1
          }


lgb_model = lgb.LGBMClassifier(**params)

counter = 1
for train_index, test_index in skf.split(train_ids, y_train):
    print('Fold {}\n'.format(counter))

    X_fit, X_val = X_train2.iloc[train_index, :], X_train2.iloc[test_index, :]
    y_fit, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

    lgb_model.fit(X_fit,
                  y_fit,
                  eval_set=[(X_val, y_val)],
                  verbose=10,
                  early_stopping_rounds=30,
                  eval_metric=evaluate_microF1_lgb
                  )

    del X_fit
    del X_val
    del y_fit
    del y_val
    del train_index
    del test_index
    gc.collect()
    
    ft_importances += lgb_model.feature_importances_

    counter += 1
    
    b_r += lgb_model.best_iteration_
    
b_r = int(b_r/k)

