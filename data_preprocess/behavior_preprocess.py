import pandas as pd
import numpy as np
import pickle
import random


def preprocess(data, negative_sampling_num, data_name):  # clicking histories on the last day as labels for training
    mat_data = []
    group_user = data.groupby('UserID')
    n = 0
    for key in group_user.groups.keys():
        if data_name == 'validation_data':
            print('generating validation data')
            if key not in validation_users:
                print(key+' is not in validation users.')
                continue
        n += 1
        print(n)
        user_list = []
        user_list.append([key])
        user_data = group_user.get_group(key)
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
        user_pre = user_data.loc[user_data['date']==max_date]
        user_pre = user_pre['Impressions'].str.split(' ')

        for i in user_pre:

            pre_true = []
            pre_false = []
            flag = 0  # count the number of positive clicks
            for j in i:
                if j.endswith('-1'):
                    flag += 1
                    pre_true.append(j.split('-')[0])
                else:
                    pre_false.append(j.split('-')[0])
            if flag != 0:  # if there's a positive click in this impression, then sample k negative samples for each positive click
                if len(pre_false) >= negative_sampling_num:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = random.sample(pre_false, negative_sampling_num)
                        li.extend(pre_false_sample)
                        user_li = user_list.copy()
                        user_li.append(li)
                        mat_data.append(user_li)
                else:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = pre_false + list(np.random.choice(pre_false, negative_sampling_num-len(pre_false)))
                        li.extend(pre_false_sample)
                        user_li = user_list.copy()
                        user_li.append(li)
                        mat_data.append(user_li)
    random.shuffle(mat_data)
    f = open(data_name+'.pkl', 'wb')
    data = {data_name: mat_data}
    pickle.dump(data, f)


if __name__ == '__main__':
    behaviors_train = pd.read_csv('../MINDlarge_train/behaviors.tsv', sep='\t', header=None)
    behaviors_train.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_train['Time'] = pd.to_datetime(behaviors_train['Time'], format='%m/%d/%Y %I:%M:%S %p')
    preprocess(behaviors_train, 4, 'training_data')

    behaviors_valid = pd.read_csv('../MINDlarge_dev/behaviors.tsv', sep='\t', header=None)
    behaviors_valid.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_valid['Time'] = pd.to_datetime(behaviors_valid['Time'], format='%m/%d/%Y %I:%M:%S %p')
    validation_users = behaviors_valid['UserID'].tolist()

    behaviors = pd.concat((behaviors_train, behaviors_valid), axis=0)
    preprocess(behaviors, 4, 'validation_data')



