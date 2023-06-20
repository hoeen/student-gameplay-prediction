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
from tqdm import tqdm
import pickle

DATA_PATH = ''
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {:.2f} sec, {:.2f} min".format(original_fn.__name__, end_time-start_time, (end_time-start_time)/60))
        return result
    return wrapper_fn


@logging_time
def main(args):
    # wandb.login()
    # wandb.init(project="gameplay_boosting", config=vars(args))

    targets = load_targets(args)
    df = pd.read_parquet(args.train)

    # remove outlier users
    # outliers = np.load(args.processed + 'outlier_users.npy')
    # df = df.set_index('session_id').drop(outliers).reset_index()

    
    # save & load models and results
    if args.level_group == '0-4':
        models = {}
        results = [[[], []] for _ in range(18)]
    else:
        models = pickle.load(open(args.model_file, 'rb'))
        results = pickle.load(open(args.result_file, 'rb'))
        

    list_q = {'0-4':[1,2,3], '5-12':[4,5,6,7,8,9,10,11,12,13], '13-22':[14,15,16,17,18]}
    # groups = ['0-4', '5-12', '13-22']

    # for grp in tqdm(groups):
    grp = args.level_group
    df_grp = df[df['level_group'] == grp]
    train, old_train = preprocessing(df_grp, grp, args)
    old_train = old_train[old_train['level_group'] == grp]
    quests = list_q[grp]
    create_model(args, train, old_train, quests, targets, models, results)
        # break ### To Test

    time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(args.model_path + time + '_lv' + grp)
    if args.cv:
        for k, q in models.keys():
            models[(k,q)].save_model(args.model_path + time + '_lv' + grp + '/' + f'{args.model}_{time}_quest{q}_fold{k}.cbm')
    else:
        for q in models.keys():
            models[q].save_model(args.model_path + time + + '_lv' + grp + '/' + f'{args.model}_{time}_quest{q}_holdout.cbm')

    # save models, results
    pickle.dump(models, open(args.model_file, 'wb'))
    pickle.dump(results, open(args.result_file, 'wb'))   

    # if last, print optimal threshold
    if args.level_group == '13-22':
        score_threshold(results)

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)

