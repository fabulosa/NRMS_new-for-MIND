import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from random import sample
from torch.autograd import Variable
import torch.utils.data as Data
import pickle
from utils import *
from NRMS_new import *
from main import pad_masking


"the difference between evaluation and validation is batch_user_valid does not contain just five news (one pos + four neg)"
"batch_user_valid need to be padded to the length of the longest sequence"


def pad_validation_news(batch_user_valid):
    "pad validation news for all users in a batch, the length should be the length of the longest sequence in the batch"
    length = [len(i) for i in batch_user_valid]
    max_len = max(length)
    batch_valid = batch_user_valid.copy()

    for i, j in enumerate(batch_valid):
        batch_valid[i].extend(['-1'] * (max_len - len(j)))

    batch_valid = np.array(list(batch_valid))
    mask = batch_valid.copy()
    mask[mask != '-1'] = 1
    mask[mask == '-1'] = 0
    mask_softmax = mask.astype(int)  # mask for softmax
    batch_valid[batch_valid == '-1'] = '0'
    return batch_valid, mask_softmax


def process_batch_data(behavior_data, index):  # padding and masking

    data = behavior_data[index]
    history = data[:, 1]
    recent = data[:, 2]
    batch_user_history, user_history_mask_selfattn, user_history_mask_attn = pad_masking(history)
    batch_user_short, user_short_mask_selfattn, user_short_mask_attn = pad_masking(recent)
    batch_user_valid, mask_softmax = pad_validation_news(data[:, 3])
    batch_user_impressionID = np.array(list(data[:, 4])).squeeze(-1)
    return batch_user_history, batch_user_short, user_history_mask_selfattn, user_history_mask_attn, user_short_mask_selfattn, user_short_mask_attn, batch_user_valid, mask_softmax, batch_user_impressionID


def evaluation(model, vali_data):

    evaluate_data_index = torch.IntTensor(np.array(range(len(evaluation_data))))
    evaluate_data_index = Data.TensorDataset(data_tensor=evaluate_data_index, target_tensor=evaluate_data_index)
    loader = Data.DataLoader(dataset=evaluate_data_index, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)

    model.eval()
    ranking = []
    impressionIDs = []
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        print(str(step)+'/'+str(num_iterations_eval))
        processed_data = process_batch_data(vali_data, batch_x.numpy())
        batch_user_history = processed_data[0]
        batch_user_short = processed_data[1]
        user_history_mask_selfattn = processed_data[2]
        user_history_mask_attn = processed_data[3]
        user_short_mask_selfattn = processed_data[4]
        user_short_mask_attn = processed_data[5]
        batch_user_valid = processed_data[6]
        batch_user_mask = processed_data[7]
        batch_user_impressionID = processed_data[8]

        # compute output
        batch_predict = model(batch_user_history, batch_user_short, user_history_mask_selfattn, user_history_mask_attn, user_short_mask_selfattn,
                            user_short_mask_attn, batch_user_valid, newsID_categoryID,
                              newsID_subcategoryID, newsID_TitleWordID, newsID_AbstractWordID, newsID_titleEntityId_conf, newsID_abstractEntityId_conf).cuda()
        batch_predict = F.softmax(batch_predict, dim=1)
        scores = batch_predict.cpu().data.numpy() * batch_user_mask
        for i in range(len(scores)):
            score = scores[i][:int(np.sum(batch_user_mask[i]))]
            rank = score.argsort().argsort() + 1
            rank = list(len(rank) + 1 - np.array(rank))
            ranking.append(rank)
        impressionIDs.extend(list(batch_user_impressionID))
    df = pd.DataFrame(list(zip(impressionIDs, ranking)), columns=['ImpressionID', 'Rank'])
    df = df.sort_values(by=['ImpressionID'], ascending=True)
    f = open(args.ranking_name, "a+")
    for i, j in df.iterrows():
        f.writelines((str(j['ImpressionID']), ' ', str(list(j['Rank'])).replace(' ', ''), '\n'))
    print('Ranking produced.')
    f.close()


if __name__ == '__main__':

    args = parse_args()
    batch_size = args.batch_size
    print('loading model')
    model = torch.load(args.model_name)

    f = open(args.newsID_categoryID, 'rb')
    newsID_categoryID = pickle.load(f)
    f = open(args.newsID_subcategoryID, 'rb')
    newsID_subcategoryID = pickle.load(f)
    f = open(args.newsID_TitleWordID, 'rb')
    newsID_TitleWordID = pickle.load(f)
    f = open(args.newsID_AbstractWordID, 'rb')
    newsID_AbstractWordID = pickle.load(f)
    f = open(args.newsID_titleEntityId_conf, 'rb')
    newsID_titleEntityId_conf = pickle.load(f)
    f = open(args.newsID_abstractEntityId_conf, 'rb')
    newsID_abstractEntityId_conf = pickle.load(f)

    print("loading evaluation data")

    f = open(args.evaluation_data, 'rb')
    evaluation_data = np.array(pickle.load(f)['evaluation_data'])

    for i, j in enumerate(evaluation_data):
        if type(j[1]) != float and len(j[1]) > 30:
            evaluation_data[i][1] = j[1][-30:]
        if type(j[2]) != float and len(j[2]) > 10:
            evaluation_data[i][2] = j[2][-10:]

    evaluation_len = len(evaluation_data)
    num_iterations_eval = evaluation_len // batch_size
    print('number of evaluated samples', evaluation_len)
    evaluation(model, evaluation_data)
