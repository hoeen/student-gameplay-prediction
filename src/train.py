import os

import torch
# import wandb
from args import parse_args
# from src import trainer
from data.dataloader import Preprocess
from utils import setSeeds


def main(args):
    # wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data() # train_data에는 user_id별로 모아놓은 데이터 존재.

    # preprocess 진행 여부 판단
    # TODO
    train_data, valid_data = preprocess.split_data(train_data)
    
    # wandb.init(project="dkt", config=vars(args))
    model = trainer.get_model(args).to(args.device)
    trainer.run(args, train_data, valid_data, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
