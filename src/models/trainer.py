import math
import os

import torch
import wandb

from tqdm import tqdm
from datetime import datetime

from data.dataloader import get_loaders, get_target

from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, DNN
from .optimizer import get_optimizer
from .scheduler import get_scheduler


def run(args, train_data, valid_data, train_target, valid_target, model):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data, train_target, valid_target)

    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_f1 = -1
    early_stopping_counter = 0
    train_time = datetime.today().strftime("%Y%m%d%H%M%S")
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_f1, train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        f1, auc, acc = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        # wandb.log(
        #     {
        #         "epoch": epoch,
        #         "train_loss_epoch": train_loss,
        #         "train_f1_epoch": train_f1,
        #         "train_auc_epoch": train_auc,
        #         "train_acc_epoch": train_acc,
        #         "valid_f1_epoch": f1,
        #         "valid_auc_epoch": auc,
        #         "valid_acc_epoch": acc,
        #     }
        # )
        if f1 > best_f1:
            best_f1 = f1
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                "model.pt",
                train_time,
                args
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_f1)


def train(train_loader, model, optimizer, scheduler, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, (batch, target) in tqdm(enumerate(train_loader)):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        preds = model(input)
        target = torch.Tensor(target).to(args.device)
        loss = compute_loss(preds, target)
        update_params(loss, model, optimizer, scheduler, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        # preds = preds[:, -1]
        # targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(target.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    f1, auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"TRAIN F1: {f1} AUC : {auc} ACC : {acc}")
    return f1, auc, acc, loss_avg

# TODO: validate with target
def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, (batch, target) in enumerate(valid_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        # predictions
        preds = model(input)

        target = torch.Tensor(target)

        total_preds.append(preds.detach())
        total_targets.append(target)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    f1, auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID F1: {f1} AUC : {auc} ACC : {acc}")

    return f1, auc, acc


def inference(args, test_data, model):

    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)

        # predictions 시작
        preds = preds[:, -1]

        ## sigmoid 추가
        preds_sig = torch.nn.Sigmoid()
        preds = preds_sig(preds)
        
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)
        # predictions 끝 

    write_path = os.path.join(args.output_dir, "submission.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "lstmattn":
        model = LSTMATTN(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == 'DNN':
        model = DNN(args)

    return model


# 배치 전처리
def process_batch(batch):

    mask = batch[-1]

    # 범주형 변수는 인코딩 0인 값을 없애기 위해 1씩 더해준다
    return (*batch[:2],  
            *map(lambda x: x.int(), [(cat+1)*mask for cat in batch[2:-1]]),
            mask
        )


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    # loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    # optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename, train_time, args):
    print(f"saving {args.model} model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, args.model + '_' + train_time + '_' + model_filename))


def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
