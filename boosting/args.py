import argparse
import platform

def parse_args():

    if platform.system() == 'Linux':
        DIR = "/home/wooseok/Python_lab/kaggle/gameplay/student-gameplay-prediction/"
    elif platform.system() == 'Darwin':
        DIR = "/Users/wooseokpark/github/kaggle/student-gameplay-prediction/"

    parser = argparse.ArgumentParser()

    parser.add_argument("--target", default=DIR + "data/raw/input/train_labels.csv", type=str, help="target csv")
    parser.add_argument("--train", default=DIR + "data/raw/input/train.parquet", type=str, help="train parquet")
    parser.add_argument("--model_path", default=DIR + "boosting/models/", type=str, help="model path")


    args = parser.parse_args()

    return args
