import os
import platform
import torch
import wandb
from args import parse_args
from models import trainer
from data.dataloader import Preprocess, get_target
from utils import setSeeds


def main(args):
    # wandb.login()

    setSeeds(args.seed)
    
    if platform.system() == 'Linux':
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif platform.system() == 'Darwin':
        args.device = "mps" if torch.cuda.is_available() else "cpu"
    
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data() # train_data에는 user_id별로 모아놓은 데이터 존재.
    
    target_data = get_target(args)

    # preprocess 진행 여부 판단
    train_data, train_target, valid_data, valid_target = preprocess.split_data(train_data, target_data)
    
    # wandb.init(project="gameplay", config=vars(args))
    model = trainer.get_model(args).to(args.device)
    trainer.run(args, train_data, valid_data, train_target, valid_target, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
