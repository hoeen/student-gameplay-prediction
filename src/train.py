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
    ## Training
    if not args.submission:
        preprocess.load_train_test_data(args.file_name, is_train=True)
    ## Inference
    else:
        preprocess.load_train_test_data(args.file_name, is_train=False)
    
    train_data, valid_data, test_data = preprocess.get_train_test_data() # train_data에는 user_id별로 모아놓은 데이터 존재.
    
    if not args.submission:
        train_target, valid_target, test_target = get_target(args, is_train=True)
    else:
        train_target, test_target = get_target(args, is_train=False)
    

    # train_data, train_target, valid_data, valid_target = preprocess.split_data(args, train_data, target_data)
    ## Training
    
    # wandb.init(project="gameplay", config=vars(args))
    model = trainer.get_model(args).to(args.device)
    

    if not args.submission:
        trainer.run(args, train_data, valid_data, train_target, valid_target, model)
        # test로 검증
        trainer.inference(test_data, test_target, model, args)
    else: # model for submission
        trainer.run(args, train_data, test_data, train_target, test_target, model)

        
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
