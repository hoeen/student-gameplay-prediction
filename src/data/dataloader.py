import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

import pickle

from utils import logging_time


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.num_cols = self.args.num_cols # ['elapsed_time', 'level']
        self.cate_cols = self.args.cate_cols # ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
        
    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, target, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        target_1 = target[:size]
        target_2 = target[size:]

        return data_1, target_1, data_2, target_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.processed_dir, name + "_classes.npy")
        np.save(le_path, encoder.categories_[0])
    
    def __save_scaler(self, scaler):
        sc_path = os.path.join(self.args.processed_dir, "cont_scaler.pkl")
        with open(sc_path, 'wb') as f:
            pickle.dump(scaler, f) # scaler 자체를 저장
    
    
    @logging_time
    def __preprocessing(self, df, is_train=True):
        
        # ordinal encoding
        for col in self.cate_cols:
            if is_train:   #train
                le = OrdinalEncoder()
                # For UNKNOWN class
                le.fit(df[[col]])
                self.__save_labels(le, col)
            else:  #inference
                label_path = os.path.join(self.args.processed_dir, col + "_classes.npy")
                le.categories_[0] = np.load(label_path, allow_pickle=True)

            test = le.transform(df[[col]])
            df[col] = test
            
        # scaling - using Robustscaling because of outliers in elapsedtime
        # self.num_cols
        if is_train:
            scaler = RobustScaler()
            scaler.fit(df[self.num_cols])
            self.__save_scaler(scaler)

        else:  #inference
            label_path = os.path.join(self.args.processed_dir, "cont_scaler.pkl")
            with open(label_path, 'rb') as f:
                scaler = pickle.load(f)
        
        df[self.num_cols] = scaler.transform(df[self.num_cols])
        

        return df

    def __feature_engineering(self, df):
        # TODO
        return df

    def load_data_from_file(self, file_name, is_train=True, processed=False):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        proc_file_path = os.path.join(self.args.processed_dir, file_name)
        
        if not processed:
            df = pd.read_parquet(csv_file_path)  # , nrows=100000)
            df = self.__feature_engineering(df)
            df = self.__preprocessing(df, is_train)
            df.to_parquet(proc_file_path)
        else:
            df = pd.read_parquet(proc_file_path)
        
        

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.cate_cols = self.cate_cols
        self.args.num_cols = self.num_cols
        for cate in self.cate_cols:
            # 'event_name', 'name', 'fqid', 'room_fqid', 'text_fqid'
            # self.args.cate 로 지정
            setattr(self.args, 
                    'input_size_'+cate, 
                    len(np.load(os.path.join(self.args.processed_dir, cate + "_classes.npy"), allow_pickle=True)) 
            )
        

        # df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["session_id"] + self.num_cols + self.cate_cols
        group = (
            df.groupby("session_id")
            .apply(
                lambda r: tuple([r[col].values for col in columns[1:]])
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name, processed=self.args.processed)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False, processed=True)


def get_target(args): # target 데이터 가져옴
    file_path = os.path.join(args.data_dir, args.target_name)
    label_df = pd.read_csv(file_path)
    # 1~18 target data by user
    # code from: https://www.kaggle.com/code/dungdore1312/session-info-as-sequence-use-lstm-to-predict/notebook
    label_df['session'] = label_df.session_id.apply(lambda x: int(x.split('_')[0]) )
    label_df['question_idx'] = label_df.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
    label_df.drop("session_id", axis=1, inplace=True)
    pivoted_questions = label_df.pivot(columns='question_idx', values='correct', index='session')
    
    return pivoted_questions.values

class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, args):
        self.data = data
        self.target = target
        self.args = args

    def __getitem__(self, index):
        row = list(self.data[index])
        target = list(self.target[index])
        # 각 data의 sequence length
        seq_len = len(row[0])

        # input 형태 : ['elapsed_time', 'level', 'event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
        # test, question, tag, correct = row[0], row[1], row[2], row[3]

        # cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len: 
            for i, col in enumerate(row):
                row[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        row.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(row):
            row[i] = torch.tensor(col)

        return row, target

    def __len__(self):
        return len(self.data)


# from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0][0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row, _ in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])
    # 실험 1 : max_seq_len (20)마다 18개의 답을 내도록 
    # 실험 2 : max_seq_len을 유저의 모든 로그 길이보다 크게 
    # 결론 : max_seq_len = 2000으로 설정하여 그 이상 데이터는 DKTDataset에서 잘라버림
    return tuple(col_list), tuple([b[1] for b in batch])

def get_loaders(args, train, valid, train_target, valid_target):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, train_target, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, valid_target, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
