# 수정
def new_page(X,revised_train, grp): 
    '''
    X= original train dataset
    '''
    # 이상치 session_id 추출
    if grp == '5-12':
        session_2=X[X.page==0].session_id.unique().tolist()
        X.loc[X['session_id'].isin(session_2), 'new_page'] = 0
        X.loc[~X['session_id'].isin(session_2), 'new_page'] = 1
        result=X.groupby(['session_id']).first().reset_index()
        result=result.set_index('session_id')['new_page']
        return pd.merge(revised_train,result, left_index=True, right_index=True, how='left')
    
    elif grp == '13-22':
        session_3=X[(X.page==0)|(X.page==1)|(X.page==2)].session_id.unique().tolist()
        X.loc[X['session_id'].isin(session_3), 'new_page'] = 0
        X.loc[~X['session_id'].isin(session_3), 'new_page'] = 1
        result=X.groupby(['session_id']).first().reset_index()
        result=result.set_index('session_id')['new_page']
        return pd.merge(revised_train,result, left_index=True, right_index=True, how='left')
    
    else:
        return revised_train



# group별 tfidf기준 correct에서 상대적으로 중요하게 나온 단어
group1_text_corr=['thing', 'fascinating', 'yes', 'been', 'he', 'wow', 'done', 'good', 'closer', 'shouldn', 'enough', 'plus', 'great', 'ugh', 'collection', 'waiting', 'room', 'catch'] 
group2_text_corr=['cough', 'like', 'grumble', 'little', 'ooh', 'mission', 'time', 'horses', 'hold', 'wayyyy', 'while', 'yikes', 'funny', 'yup', 'sore', 'kid', 'letters', 'tiny', 'pony', 'missions', 'throat', 'blasted', 'two', 'actually', 'horse', 'or', 'hoarse', 'huh', 'ransacked', 'doesn', 'been', 'then', 'kidnapped', 'really', 'drop', 'something', 'dear', 'card', 'floor'] 
group1_text_wron=['undefined', 'look', 'did', 'clues', 'back', 'of', 'best', 'later', 'grampa', 'them', 'ooh', 'bee', 'knees', 'love', 'photos', 'these', 'button', 'ever', 'better', 'with', 'check', 'hopefully', 'um', 'want', 'again', 'because', 'couldn', 'those', 'hooray', 'never', 'works', 'play', 'should', 'feel', 'forgetting', 'something', 'around']
group2_text_wron=['on', 'archivist', 'should', 'go', 'out', 'stacks', 'haven', 'any', 'talk', 'university', 'well', 'figure', 'cleaning', 'so', 'capitol', 'wearing', 'check', 'better', 'upstairs', 'badgers', 'undefined', 'still', 'work', 'love', 'these', 'photos', 'button', 'jolie', 'poor', 'make', 'hear', 'loose', 'museum', 'lost', 'glasses', 'believe', 'reading', 'yeah', 'head', 'again', 'along', 'run', 'counting', 'yet', 'quite', 'figured', 'hoping', 'glass', 'u00e2', 'by', 'used', 'magnifying', 'news', 'u20ac', 'stop', 'u00a6', 'artifact', 'much', 'working', 'hmm', 'works', 'over', 'waiting', 'clue', 'clean', 'might', 'able', 'said', 'somewhere', 'weren', 'going', 'later', 'nice', 'some', 'libarian', 'information', 'seeing', 'decorations', 'such', 'day', 'fall']
group3_text_wron=['any', 'artifact', 'by', 'if', 'right', 'digging', 'were', 'library', 'well', 'figured', 'quite', 'figure', 'stacks', 'able', 'badgers', 'friend', 'outside', 'haven', 'hoping', 'stop', 'news', 'yet', 'counting', 'door', 'they', 'funny', 'lynx', 'info', 'book', 'anyway', 'newspapers', 'loaded', 'slowing', 'fall', 'such', 'later', 'seeing', 'luck', 'ready', 'when', 'hear', 'loose', 'wearing', 'shirt', 'youmans']


def add_text(data, revised_data, col, word_list, col_name_mean, col_name_std, col_name_max):
    '''
    data= data set ex. df1, df2, df3
    col= 수정하고 싶은 열이름 str
    word_list = tfidf relative_diff (word_list)
    col_name: 새로 만들고 싶은 열이름 str
    '''
    # Create a boolean mask for rows containing the specific words
    word_mask = data[col].str.contains('|'.join(word_list), case=False)
    word_mask.fillna(False, inplace=True)
    
    # Group the data by 'session_id' and calculate the mean of 'time_diff' for the selected rows
    result1=data[word_mask].groupby('session_id')['elapsed_time_diff'].mean()
    result1.reindex(data['session_id'], fill_value=0)
    result2=data[word_mask].groupby('session_id')['elapsed_time_diff'].std()
    result2.reindex(data['session_id'], fill_value=0)
    result3=data[word_mask].groupby('session_id')['elapsed_time_diff'].max()
    result3.reindex(data['session_id'], fill_value=0)
    
    mean_=pd.DataFrame({col_name_mean:result1})
    std_=pd.DataFrame({col_name_std:result2})
    max_=pd.DataFrame({col_name_max:result3})

    total= pd.concat([mean_,std_, max_], axis=1)

    return pd.merge(revised_data,total, left_index=True, right_index=True, how='left')

  
   def playtime(data):
    tmp_df = []
    #NEW

    qvant = data.groupby(["session_id", "level_group"])['elapsed_time_diff'].quantile(q=0.3)
    qvant.name = 'qvant1_0_3'
    tmp_df.append(qvant)

    qvant = data.groupby(["session_id", "level_group"])['elapsed_time_diff'].quantile(q=0.8)
    qvant.name = 'qvant2_0_8'
    tmp_df.append(qvant)

    qvant = data.groupby(["session_id", "level_group"])['elapsed_time_diff'].quantile(q=0.5)
    qvant.name = 'qvant3_0_5'
    tmp_df.append(qvant)

    qvant = data.groupby(["session_id", "level_group"])['elapsed_time_diff'].quantile(q=0.65)
    qvant.name = 'qvant4_0_65'
    tmp_df.append(qvant)
    
    #data.drop(EVENT, axis = 1, inplace =True) # 將上面做的獨立出來的Event欄位刪除
        
    # "elapsed_time" 單獨計算每個level_group所花的時間   
    tmp = data.groupby(["session_id", "level_group"])["elapsed_time"].apply(lambda x: x.max() - x.min())
    tmp.name = "playtime" #此關卡所用的時間
    tmp_df.append(tmp)
        
    df = pd.concat(tmp_df, axis = 1)
    df = df.fillna(-1) ##?????????????????????
    df = df.reset_index() #將sesion_id、level_group 從index拉回df column
    df = df.set_index("session_id")
    df.drop('level_group', axis=1,inplace=True)
    return df


  
  
  
    def preprocessing(df, grp):
    start, end = map(int,grp.split('-'))
    kol_lvl = (df.groupby(['session_id'])['level'].agg('nunique') < end - start + 1)
    list_session = kol_lvl[kol_lvl].index
    df = df[~df['session_id'].isin(list_session)]
    df = delt_time_def(df)
    train_ = feature_engineer(pl.from_pandas(df), grp, use_extra=False, feature_suffix='')
    # recap text count \w join
    train = text_cnt(df, train_)
    train['recap_reading'] = train['recap_reading'].fillna(0)
    # add year, month, day etc.
    train = time_feature(train)
    # new_page
    train = new_page(df, train, grp)
    # Playtime
    pla=playtime(df)
    train = pd.merge(train, pla, left_index=True, right_index=True, how='left')

    # add_text
    if grp =='0-4':
        train=add_text(df, train, 'text', group1_text_corr, 'corr_mean', 'corr_std', 'corr_max')
        train=add_text(df, train, 'text', group1_text_wron, 'wron_mean', 'wron_std', 'wron_max' )
        return train, df

    elif grp=='5-12':
        train= add_text(df, train, 'text', group2_text_corr, 'corr_mean', 'corr_std', 'corr_max' )
        train= add_text(df, train, 'text', group2_text_wron, 'wron_mean', 'wron_std', 'wron_max' )
        return train, df

    else:
        train=add_text(df, train, 'text', group3_text_wron, 'wron_mean', 'wron_std', 'wron_max' ) 
        return train, df
