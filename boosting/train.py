from args import parse_args
from feature_engineering import *
from model import create_model
from evaluate import score_threshold

import pandas as pd
import polars as pl

import os
from datetime import datetime
import wandb
import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


@logging_time
def main(args):
    # wandb.login()
    # wandb.init(project="gameplay_boosting", config=vars(args))

    targets = load_targets(args)
    df = pd.read_parquet(args.train)
    
    # concat additional data
    df, targets = add_data(df, targets)

    # remove outlier users
    # outliers = np.load(args.processed + 'outlier_users.npy')
    # df = df.set_index('session_id').drop(outliers).reset_index()

    models = {}
    results = [[[], []] for _ in range(18)]
    list_q = {'0-4':[1,2,3], '5-12':[4,5,6,7,8,9,10,11,12,13], '13-22':[14,15,16,17,18]}
    groups = ['0-4', '5-12', '13-22']

    for grp in groups:
        df_grp = df[df['level_group'] == grp]
        train, old_train = preprocessing(df_grp, grp)
        old_train = old_train[old_train['level_group'] == grp]
        quests = list_q[grp]
        create_model(args, train, old_train, quests, targets, models, results)
        # break ### To Test

    score_threshold(results)

    time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(args.model_path + time)
    if args.cv:
        for k, q in models.keys():
            models[(k,q)].save_model(args.model_path + time + '/' + f'catboost_{time}_quest{q}_fold{k}.cbm')
    else:
        for q in models.keys():
            models[q].save_model(args.model_path + time + '/' + f'catboost_{time}_quest{q}_holdout.cbm')
        
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)

