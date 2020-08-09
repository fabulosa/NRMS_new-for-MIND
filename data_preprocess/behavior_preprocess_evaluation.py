import pandas as pd
import numpy as np
import pickle
import random


def generate_evaluation_data(data):  # clicking histories on the last day as labels for training
    mat_data = []
    group_user = data.groupby('UserID')
    n = 0
    for key in group_user.groups.keys():
        n += 1
        print(n)
        user_list = []
        user_list.append([key])
        user_data = group_user.get_group(key)
        #print(user_data)
        user_long_his = user_data['Histories'].str.split(' ').iloc[0]
        user_list.append(user_long_his)
        # sort by timestamp
        user_data = user_data.sort_values(['Time'], ascending=True)
        user_data['date'] = user_data['Time'].apply(lambda x: x.date())
        max_date = user_data['date'].tolist()[-1]
        user_short_his = user_data.loc[user_data['date']!=max_date]
        user_short_his = user_short_his['Impressions'].str.split(' ')
        short_his = []
        for i in user_short_his:
            for j in i:
                if j.endswith('-1'):
                    short_his.append(j.split('-')[0])
        user_list.append(short_his)
        user_pre = user_data.loc[user_data['date'] == max_date]
        # check whether in validation set

        for index, i in user_pre.iterrows():

            if i['label'] != 1:
                continue
            else:
                user_pre_each = i['Impressions'].split(' ')
                pre = []

                for j in user_pre_each:
                    pre.append(j.split('-')[0])

                user_li = user_list.copy()
                user_li.append(pre)
                user_li.append([i['ImpressionID']])
                mat_data.append(user_li)
    f = open('evaluation_data.pkl', 'wb')
    data = {'evaluation_data': mat_data}
    pickle.dump(data, f)
    f.close()


if __name__ == '__main__':
    behaviors_train = pd.read_csv('MINDlarge_train/behaviors.tsv', sep='\t', header=None)
    behaviors_train.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_train['Time'] = pd.to_datetime(behaviors_train['Time'], format='%m/%d/%Y %I:%M:%S %p')
    behaviors_train['label'] = 0

    behaviors_valid = pd.read_csv('MINDlarge_dev/behaviors.tsv', sep='\t', header=None)
    behaviors_valid.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_valid['Time'] = pd.to_datetime(behaviors_valid['Time'], format='%m/%d/%Y %I:%M:%S %p')
    behaviors_valid['label'] = 1
    impressionid = behaviors_valid['ImpressionID'].tolist()

    behaviors = pd.concat((behaviors_train, behaviors_valid), axis=0)
    generate_evaluation_data(behaviors_valid)



