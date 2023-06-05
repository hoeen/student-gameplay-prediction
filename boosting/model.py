from feature_engineering import feature_quest
from catboost import CatBoostClassifier

def create_model(train, old_train, quests, targets, models: dict):
    kol_quest = len(quests)
    ALL_USERS = train.index.unique()
    print('We will train with', len(ALL_USERS) ,'users info')
   
    print('### quest', end='')
    # ITERATE THRU QUESTIONS
    for q in quests:
        print('# ', q, end='')
        
        train_q = feature_quest(train, q)
        
        # TRAIN DATA
        train_x = train_q
        train_users = train_x.index.values
        train_y = targets.loc[targets.q==q].set_index('session').loc[train_users]

        # TRAIN MODEL 

        model = CatBoostClassifier(
            n_estimators = 300,
            learning_rate= 0.045,
            depth = 6
        )
        
        model.fit(train_x, train_y['correct'], verbose=False)

        # SAVE MODEL, PREDICT VALID OOF
       
        models[f'{q}'] = model
    print('***')
    
    return models