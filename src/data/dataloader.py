import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import torch
import tqdm
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split

import pickle

from utils import logging_time


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.num_cols = self.args.num_cols # ['elapsed_time', 'level']
        self.cate_cols = self.args.cate_cols # ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
        
    def get_train_test_data(self):
        return self.train_data, self.eval_data, self.test_data

    def split_data(self, args, data, target, ratio=0.8, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)


        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:size + (len(data)-size)//2]
        test_data = data[size + (len(data)-size)//2:] # eval 을 둘로나눠 eval, test로 함

        target_1 = target[:size]
        target_2 = target[size:size + (len(data)-size)//2]
        test_target= target[size + (len(data)-size)//2:]

        args.test_data = test_data
        args.test_target = test_target
        return data_1, target_1, data_2, target_2

    def __save_labels(self, encoder):
        for col_idx in range(len(self.cate_cols)):
            le_path = os.path.join(self.args.processed_dir, self.cate_cols[col_idx] + "_classes_"+str(self.args.level_group)+".npy")
            np.save(le_path, encoder.categories_[col_idx])
    
    def __save_scaler(self, scaler):
        sc_path = os.path.join(self.args.processed_dir, "cont_scaler_"+str(self.args.level_group)+".pkl")
        with open(sc_path, 'wb') as f:
            pickle.dump(scaler, f) # scaler 자체를 저장

    def __save_encoder(self, encoder):
        sc_path = os.path.join(self.args.processed_dir, "ord_encoder_"+str(self.args.level_group)+".pkl")
        with open(sc_path, 'wb') as f:
            pickle.dump(encoder, f) # scaler 자체를 저장
    
    
    @logging_time
    def __preprocessing(self, df, is_train=True):
        # train-eval split
        train_idx, eval_idx = train_test_split(df['session_id'].unique(), train_size=self.args.train_size)
        eval_idx, test_idx = train_test_split(eval_idx, train_size=0.5) # 0.1 / 0.1 로 eval, test 나눔

        if is_train:
            train_df = df.set_index('session_id').loc[train_idx].reset_index()
            eval_df = df.set_index('session_id').loc[eval_idx].reset_index()
            test_df = df.set_index('session_id').loc[test_idx].reset_index()
            le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # For UNKNOWN class
            le.fit(train_df[self.cate_cols])
            self.__save_encoder(le)
            self.__save_labels(le)
            
             # scaling - using Robustscaling because of outliers in elapsedtime
            scaler = RobustScaler()
            scaler.fit(train_df[self.num_cols])
            self.__save_scaler(scaler)
            
            train_df[self.cate_cols] = le.transform(train_df[self.cate_cols])
            train_df[self.num_cols] = scaler.transform(train_df[self.num_cols])
            
            eval_df[self.cate_cols] = le.transform(eval_df[self.cate_cols])
            eval_df[self.num_cols] = scaler.transform(eval_df[self.num_cols])

            test_df[self.cate_cols] = le.transform(test_df[self.cate_cols])
            test_df[self.num_cols] = scaler.transform(test_df[self.num_cols])
            
            return train_df, eval_df, test_df

        else: # inference
            train_df = df.set_index('session_id').loc[np.concatenate([train_idx, eval_idx])].reset_index()
            test_df = df.set_index('session_id').loc[test_idx].reset_index()

            encoder_path = os.path.join(self.args.processed_dir, "ord_encoder_"+str(self.args.level_group)+".pkl")
            with open(encoder_path, 'rb') as f:
                le = pickle.load(f)
            
            label_path = os.path.join(self.args.processed_dir, "cont_scaler_"+str(self.args.level_group)+".pkl")
            with open(label_path, 'rb') as f:
                scaler = pickle.load(f)

            train_df[self.cate_cols] = le.transform(train_df[self.cate_cols])
            train_df[self.num_cols] = scaler.transform(train_df[self.num_cols])
            
            test_df[self.cate_cols] = le.transform(test_df[self.cate_cols])
            test_df[self.num_cols] = scaler.transform(test_df[self.num_cols])
            
            return train_df, None, test_df

    def load_data_from_file(self, file_name, is_train=True, processed=False):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        proc_train_path = os.path.join(self.args.processed_dir, 'train_group' + str(self.args.level_group) + '.parquet')
        proc_eval_path = os.path.join(self.args.processed_dir, 'eval_group' + str(self.args.level_group) + '.parquet')
        proc_train_inf_path = os.path.join(self.args.processed_dir, 'train_inf_group' + str(self.args.level_group) +'.parquet')
        proc_test_path = os.path.join(self.args.processed_dir, 'test_group' + str(self.args.level_group) + '.parquet')
        
        # 훈련 - train, eval / 추론 - train_inf, test
        if not processed:
            df = pd.read_parquet(csv_file_path)  # , nrows=100000)
            df = feature_engineering(self.args, df, which='data')
            train_df, eval_df, test_df = self.__preprocessing(df, is_train)
            if is_train:
                train_df.to_parquet(proc_train_path)
                eval_df.to_parquet(proc_eval_path)
                test_df.to_parquet(proc_test_path)
            else:
                train_df.to_parquet(proc_train_inf_path)
                test_df.to_parquet(proc_test_path)
        
        else:
            if is_train:
                train_df = pd.read_parquet(proc_train_path)
                eval_df = pd.read_parquet(proc_eval_path)
                test_df = pd.read_parquet(proc_test_path)
            else:
                train_df = pd.read_parquet(proc_train_inf_path)
                test_df = pd.read_parquet(proc_test_path)



        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.cate_cols = self.cate_cols
        self.args.num_cols = self.num_cols
        for cate in self.cate_cols:
            # 'event_name', 'name', 'fqid', 'room_fqid', 'text_fqid'
            # self.args.cate 로 지정
            setattr(self.args, 
                    'input_size_'+cate, 
                    len(np.load(os.path.join(self.args.processed_dir, cate + "_classes_"+str(self.args.level_group)+".npy"), allow_pickle=True)) 
            )
        if is_train:
            return train_df, eval_df, test_df
        else:
            return train_df, None, test_df
    
    def groupby_session(self, df):
        # df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["session_id"] + self.num_cols + self.cate_cols
        group = (
            df.groupby("session_id")
            .apply(
                lambda r: tuple([r[col].values for col in columns[1:]])
                # lambda r: tuple([r[col].values[:0] for col in columns[1:]])
            )
        )
        return group.values

    def load_train_test_data(self, file_name, is_train=True):
        train_df, eval_df, test_df = self.load_data_from_file(file_name, is_train=is_train, processed=self.args.processed)
        self.train_data = self.groupby_session(train_df)
        self.test_data = self.groupby_session(test_df)
        # save to args
        self.args.train_session = train_df['session_id'].values
        self.args.test_session = test_df['session_id'].values
        if is_train:
            self.eval_data = self.groupby_session(eval_df)
            self.args.eval_session = eval_df['session_id'].values
        else:
            self.eval_data = None

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False, processed=True)

def feature_engineering(args, df, which='data'): # which : data / label
    # 선택한 level에 따른 data만 가져오기
    group_dic = {
        1: ('0-4', '1-3'), 
        2: ('5-12', '4-13'),
        3: ('13-22', '14-18')
    }
    group_level, question_range = group_dic[args.level_group]
    if which == 'data':
        df['elapsed_time'] = df.groupby('session_id')['elapsed_time'].apply(lambda x: x - x.shift(1))
        df = df.loc[df.level_group == group_level]
        df['elapsed_time'] = df['elapsed_time'].fillna(0)
    else:
        qstart, qend = map(int, question_range.split('-'))
        df = df.loc[df.question_idx.isin(range(qstart, qend+1))]
    
    return df

def get_target(args, is_train=True): # target 데이터 가져옴
    file_path = os.path.join(args.data_dir, args.target_name)
    # using polars to boost the speed
    label_df = pd.read_csv(file_path)
    # 1~18 target data by user
    # code from: https://www.kaggle.com/code/dungdore1312/session-info-as-sequence-use-lstm-to-predict/notebook
    label_df['session'] = label_df.session_id.apply(lambda x: int(x.split('_')[0]) )
    label_df['question_idx'] = label_df.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )

    label_df = feature_engineering(args, label_df, which='label')

    # train, test 따로 가져오기
    label_pl = pl.DataFrame(label_df)
    train_target = label_pl.filter(pl.col('session').is_in(args.train_session)).to_pandas()
    test_target = label_pl.filter(pl.col('session').is_in(args.test_session)).to_pandas()

    train_target.drop("session_id", axis=1, inplace=True)
    test_target.drop("session_id", axis=1, inplace=True)

    train_target = train_target.pivot(columns='question_idx', values='correct', index='session')
    test_target = test_target.pivot(columns='question_idx', values='correct', index='session')
    
    if not is_train:
        return train_target.values, test_target.values
    else:
        eval_target = label_pl.filter(pl.col('session').is_in(args.eval_session)).to_pandas()
        eval_target.drop("session_id", axis=1, inplace=True)
        eval_target = eval_target.pivot(columns='question_idx', values='correct', index='session')
        return train_target.values, eval_target.values, test_target.values

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
    # max_seq_len = 2000으로 설정하여 그 이상 데이터는 DKTDataset에서 잘라버림
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
