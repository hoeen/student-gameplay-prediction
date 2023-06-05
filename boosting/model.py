from feature_engineering import feature_quest
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np

def create_model(train, old_train, quests, targets, models: dict, results: list):
    kol_quest = len(quests)
    cate_cols = train.dtypes[train.dtypes == 'object'].index.tolist()
    # ALL_USERS = train.index.unique()
    # print('We will train with', len(ALL_USERS) ,'users info')
    print(f'Using {len(train.columns)} columns')
    # ITERATE THRU QUESTIONS
    for q in quests:        
        train_q = feature_quest(train, q)
        
        # TRAIN DATA
        train_x = train_q
        train_users = train_x.index.values
        train_y = targets.loc[targets.q==q].set_index('session').loc[train_users]

        # TRAIN MODEL - CV
        gkf = GroupKFold(n_splits=5)
        f1_list, precision_list, recall_list = [], [], []
        print('Fold:', end= '')
        for k, (train_idx, val_idx) in enumerate(gkf.split(train_x, groups = train_users)):
            print(k+1, end=' ')
            
            X_train = train_x.iloc[train_idx]
            X_val = train_x.iloc[val_idx]

            y_train = train_y.iloc[train_idx]['correct']
            y_val = train_y.iloc[val_idx]['correct'].values

            model = CatBoostClassifier(
                # n_estimators = 300,
                # learning_rate= 0.045,
                # depth = 6,
                devices='GPU',
                # n_estimators=1, depth=1
            )
            
            model.fit(X_train, y_train, verbose=False, 
                    cat_features = cate_cols)
            
            # SAVE MODEL
            models[(k, q)] = model #fold, q

            y_pred = model.predict_proba(X_val)[:,1]
            
            # scores
            f1 = f1_score(y_val, y_pred > 0.5, average='macro')
            precision = precision_score(y_val, y_pred > 0.5)
            recall = recall_score(y_val, y_pred > 0.5)
            f1_list.append(f1); precision_list.append(precision); recall_list.append(recall)
        print()
        print(f'Question {q} - Scores after {k+1} fold: F1: {np.mean(f1_list):.5f} Precision: {np.mean(precision_list):.5f} Recall: {np.mean(recall_list):.5f}')
        results[q - 1][0].append(y_val)
        results[q - 1][1].append(y_pred)
        
    
    return