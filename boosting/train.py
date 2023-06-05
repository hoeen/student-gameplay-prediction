from args import parse_args
from feature_engineering import *
from model import create_model
from evaluate import score_threshold

import pandas as pd
import polars as pl

import os
from datetime import datetime
import wandb

def main(args):
    # wandb.login()
    # wandb.init(project="gameplay_boosting", config=vars(args))

    targets = load_targets(args)
    df = pd.read_parquet(args.train)
    models = {}
    results = [[[], []] for _ in range(18)]
    list_q = {'0-4':[1,2,3], '5-12':[4,5,6,7,8,9,10,11,12,13], '13-22':[14,15,16,17,18]}
    groups = ['0-4', '5-12', '13-22']

    for grp in groups:
        df_grp = df[df['level_group'] == grp]
        train = preprocessing(df_grp, grp)
        quests = list_q[grp]
        create_model(train, df_grp, quests, targets, models, results)

    score_threshold(results)

    time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(args.model_path + time)
    for k, q in models.keys():
        models[(k,q)].save_model(args.model_path + time + '/' + f'catboost_{time}_quest{q}_fold{k}.cbm')
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)

