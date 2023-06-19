import argparse
import platform

def parse_args():

    if platform.system() == 'Linux':
        DIR = "/home/wooseok/Python_lab/kaggle/gameplay/student-gameplay-prediction/"
    elif platform.system() == 'Darwin':
        DIR = "/Users/wooseokpark/github/kaggle/student-gameplay-prediction/"

    parser = argparse.ArgumentParser()

    # parser.add_argument("--target", default=DIR + "data/raw/input/train_labels.csv", type=str, help="target csv")
    parser.add_argument("--target", default=DIR + "data/processed/concat/total_target_full_labeled.parquet", type=str, help="target csv")
    # parser.add_argument("--train", default=DIR + "data/raw/input/train.parquet", type=str, help="train parquet")
    parser.add_argument("--train", default=DIR + "data/processed/concat/total_train_fully_labeled.parquet", type=str, help="train parquet")
    parser.add_argument("--model_path", default=DIR + "boosting/models/", type=str, help="model path")
    parser.add_argument("--cv", default=1, type=int, help="using cross-validation")
    # parser.add_argument("--nullcol", default=DIR + "boosting/processed/null_feat.npy", type=str, help="null cols")
    parser.add_argument("--processed", default=DIR + "boosting/processed/", type=str, help="processed path")
    parser.add_argument("--model", default='xgb', type=str, help='which model')
    parser.add_argument("--model_file", default=DIR + "boosting/processed/model.pkl", type=str)
    parser.add_argument("--result_file", default=DIR + "boosting/processed/result.pkl", type=str)
    parser.add_argument("--level_group", default="0-4", type=str)
    args = parser.parse_args()

    return args
