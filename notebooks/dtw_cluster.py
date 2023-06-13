import pandas as pd
import numpy as np
import random
random.seed(20)
import pickle
import warnings
import matplotlib.pyplot as plt

# using dynamic time warping (DTW) distance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)

BASIC_PATH = '/home/wooseok/Python_lab/kaggle/gameplay/student-gameplay-prediction/'
INPUT_PATH = 'data/raw/input/'

def label_cluster(data, room, which='room'): 
    
    print(f'clustering paths in {room} - {which} data')
    
    xy = list(zip(
        data.loc[data['room_fqid'] == room, which+'_coor_x'],
        data.loc[data['room_fqid'] == room, which+'_coor_y']
    ))

    series = [np.array(d) for d in [list(zip(*r)) for r in xy]]

    n_clusters_max = 20
    inertia_values = [] 
    best_model = [None, None, None] # n_cluster 지정시 바로 나올수있게 인덱스 맞춤
    # best_clusters = 0
    min_inertia = float('inf')

    # sample 2000 data to fit
    print('n_clusters:', end=' ')
    for n_clusters in range(3, n_clusters_max + 1):
        print(n_clusters, end=' ')
        tkm = TimeSeriesKMeans(
            n_clusters=n_clusters, 
            metric="dtw", 
            max_iter=100, 
            verbose=False, 
            random_state=42, 
            n_jobs=6
            )
        tkm.fit(to_time_series_dataset(random.sample(series, 2000)).astype('int'))
        inertia = tkm.inertia_
        inertia_values.append(inertia)
        best_model.append(tkm)

    print('...all range covered')
    ## plot to choose manually after visual inspection
    plt.figure()
    plt.plot(range(3, n_clusters_max + 1), inertia_values, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Scree Plot")
    plt.ylim(min(inertia_values)-100000, int(1e9))
    plt.show()
    print('inertia_values:', inertia_values)
    best_n_cluster = int(input("Choose best n_cluster: "))
    print('use best model - n_clusters:', best_n_cluster)

    prediction = best_model[best_n_cluster].predict(to_time_series_dataset(series).astype('int'))

    return prediction, best_model[best_n_cluster] 


train = pd.read_parquet(BASIC_PATH + INPUT_PATH + 'train.parquet')

room_df = train.loc[:, ['room_coor_x', 'room_coor_y']]

screen_df = train.loc[:, ['screen_coor_x', 'screen_coor_y']]

cx_room = train.groupby(['session_id','room_fqid'])['room_coor_x'].apply(list)
cy_room = train.groupby(['session_id','room_fqid'])['room_coor_y'].apply(list)

cx_screen = train.groupby(['session_id','room_fqid'])['screen_coor_x'].apply(list)
cy_screen = train.groupby(['session_id','room_fqid'])['screen_coor_y'].apply(list)

cx_cy_room = pd.concat([cx_room, cy_room],axis=1).reset_index().set_index('session_id')
cx_cy_screen = pd.concat([cx_screen, cy_screen], axis=1).reset_index().set_index('session_id')

# result dataframe to save
result = pd.DataFrame(train.session_id.unique(), columns=['session_id'])

room_ids = train.room_fqid.unique()
clf_dict = {}
for room in room_ids:
    clf_room, room_model = label_cluster(cx_cy_room, room, 'room')
    clf_screen, screen_model = label_cluster(cx_cy_screen, room, 'screen')
    # save model
    clf_dict[room] = (room_model, screen_model)
    result[f'clf_room_path_{room}'] = clf_room
    result[f'clf_screen_path_{room}'] = clf_screen

with open(BASIC_PATH + 'models/dtw_clf_dict.pkl', 'wb') as f:
    pickle.dump(clf_dict, f)

# save labels
result.to_csv(BASIC_PATH + 'data/processed/dtw_label.csv', index=None)




