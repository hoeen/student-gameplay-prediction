from feature_engineering import feature_quest
# from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from xgboost import XGBClassifier

import numpy as np
import pickle

xgb_params = {
        'booster': 'gbtree',
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric':'logloss',
        'learning_rate': 0.02,
        'alpha': 8,
        'max_depth': 4,
        'subsample':0.8,
        'colsample_bytree': 0.5,
        'seed': 42
        }

estimators_xgb = [498, 448, 378, 364, 405, 495, 456, 249, 384, 405, 356, 262, 484, 381, 392, 248 ,248, 345]

def create_model(args, train, old_train, quests, targets, models: dict, results: list):
    kol_quest = len(quests)
    cate_cols = train.dtypes[train.dtypes == 'object'].index.tolist()
    # ALL_USERS = train.index.unique()
    # print('We will train with', len(ALL_USERS) ,'users info')
    if args.cv:
        print('using CV...')
    else:
        print('using hold-out...')

    ### high null columns - experimental ###
    # with open(args.nullcol, 'rb') as f:
    #     null_feat = pickle.load(f)

    # if quests[0] == 1:
    #     nkey = '0-4'
    # elif quests[0] == 4:
    #     nkey = '5-12'
    # elif quests[0] == 14:
    #     nkey = '13-22'
    
    
    # null_cols = null_feat[nkey]

    # train = train.loc[:, [col for col in train.columns if col not in null_cols]]
    # old_train = old_train.loc[:, [col for col in old_train.columns if col not in null_cols]]

    print(f'Using {len(train.columns)} columns')
    
    # ITERATE THRU QUESTIONS
    for q in quests:   
        print('Question', q)     
        train_q = feature_quest(train, old_train, q)
        
        # set n_estimator params
        # from : https://www.kaggle.com/code/pourchot/simple-xgb
        xgb_params['n_estimators'] = estimators_xgb[q-1]

        # TRAIN DATA
        train_x = train_q
        train_users = train_x.index.values
        train_y = targets.loc[targets.q==q].set_index('session').loc[train_users]

        # TRAIN MODEL - CV
        if args.cv:
            gkf = GroupKFold(n_splits=5)
            f1_list, precision_list, recall_list = [], [], []
            print('Fold:', end= '')
            for k, (train_idx, val_idx) in enumerate(gkf.split(train_x, groups = train_users)):
                print(k+1, end=' ')
                
                X_train = train_x.iloc[train_idx]
                X_val = train_x.iloc[val_idx]

                y_train = train_y.iloc[train_idx]['correct']
                y_val = train_y.iloc[val_idx]['correct'].values
                
                model = XGBClassifier(
                    **xgb_params
                )
                
                model.fit(X_train, y_train, verbose=True,)
                        # cat_features = cate_cols)
                
                # SAVE MODEL
                models[(k, q)] = model #fold, q

                y_pred = model.predict_proba(X_val)[:,1]
                
                # scores
                f1 = f1_score(y_val, y_pred > 0.5, average='macro')
                precision = precision_score(y_val, y_pred > 0.5)
                recall = recall_score(y_val, y_pred > 0.5)
                f1_list.append(f1); precision_list.append(precision); recall_list.append(recall)

                results[q - 1][0].append(y_val)
                results[q - 1][1].append(y_pred)

            print()
            print(f'Question {q} - Scores after {k+1} fold: F1: {np.mean(f1_list):.5f} Precision: {np.mean(precision_list):.5f} Recall: {np.mean(recall_list):.5f}')
            

        else: # hold-out 
            user_train, user_val = train_test_split(train_x.index.values, random_state=42)
            X_train = train_x.loc[user_train]
            X_val = train_x.loc[user_val]

            y_train = train_y.loc[user_train]['correct']
            y_val = train_y.loc[user_val]['correct'].values

            model = XGBClassifier(
                    **xgb_params
                )
            
            model.fit(X_train, y_train, verbose=False,) 
                    # cat_features = cate_cols)
            
            # SAVE MODEL
            models[q] = model #fold, q

            y_pred = model.predict_proba(X_val)[:,1]
            
            # scores
            f1 = f1_score(y_val, y_pred > 0.5, average='macro')
            precision = precision_score(y_val, y_pred > 0.5)
            recall = recall_score(y_val, y_pred > 0.5)
            print(f'Question {q} - Scores F1: {f1:.5f} Precision: {precision:.5f} Recall: {recall:.5f}')
            results[q - 1][0].append(y_val)
            results[q - 1][1].append(y_pred)

        # break  #### to test

    return


