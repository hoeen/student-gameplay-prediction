from args import parse_args
from feature_engineering import *
from model import create_model

import pandas as pd
import polars as pl

from datetime import datetime

def main(args):

    targets = load_targets(args)
    df = pd.read_parquet(args.train)
    models = {}
    list_q = {'0-4':[1,2,3], '5-12':[4,5,6,7,8,9,10,11,12,13], '13-22':[14,15,16,17,18]}
    groups = ['0-4', '5-12', '13-22']

    for grp in groups:
        df_grp = df[df['level_group'] == grp]
        train = preprocessing(df_grp, grp)
        quests = list_q[grp]
        create_model(train, df_grp, quests, targets, models)
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    for k in models.keys():
        models[k].save(args.model_path + f'catboost_{time}_q{k}.cbm')
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)

