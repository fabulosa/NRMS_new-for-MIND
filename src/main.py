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


def pad_masking(bat):
    "padding and masking news for batch"
    length = [len(i) if type(i)!=float else 0 for i in bat]
    max_len = max(length)
    batch = bat.copy()
    if max_len == 0:
        for i, j in enumerate(batch):
            max_len = 1
            batch[i] = ['-1'] * max_len

    for i, j in enumerate(batch):
        if type(j) != float:
            batch[i].extend(['-1'] * (max_len - len(j)))
        else:
            batch[i] = ['-1'] * max_len

    batch = np.array(list(batch))
    mask = batch.copy()
    mask[mask != '-1'] = 1
    mask[mask == '-1'] = 0
    mask = mask.astype(int)  # mask for additive attention
    mask_attn = mask.copy()
    mask1 = mask[:, :, np.newaxis]  # mask for self attention
    mask2 = mask[:, np.newaxis, :]
    mask = np.matmul(mask1, mask2)
    batch[batch == '-1'] = '0'
    return batch, torch.IntTensor(mask), torch.IntTensor(mask_attn).cuda()


def process_batch_data(behavior_data, index):  # padding and masking
    data = behavior_data[index]
    history = data[:, 1]
    recent = data[:, 2]
    batch_user_history, user_history_mask_selfattn, user_history_mask_attn = pad_masking(history)
    batch_user_short, user_short_mask_selfattn, user_short_mask_attn = pad_masking(recent)
    batch_user_valid = np.array(list(data[:, 3])).squeeze(-1)
    batch_label = np.array(list(data[:, 4])).squeeze(-1)
    return batch_user_history, batch_user_short, user_history_mask_selfattn, user_history_mask_attn, user_short_mask_selfattn, user_short_mask_attn, batch_user_valid, batch_label


def evaluate(model, loader, vali_data):
    model.eval()
    summ = []
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        processed_data = process_batch_data(vali_data, batch_x.numpy())
        batch_user_history = processed_data[0]
        batch_user_short = processed_data[1]
        user_history_mask_selfattn = processed_data[2]
        user_history_mask_attn = processed_data[3]
        user_short_mask_selfattn = processed_data[4]
        user_short_mask_attn = processed_data[5]
        batch_user_valid = processed_data[6]
        batch_label = processed_data[7]
        # compute output
        batch_predict = model(batch_user_history, batch_user_short, user_history_mask_selfattn, user_history_mask_attn, user_short_mask_selfattn, user_short_mask_attn, batch_user_valid, newsID_categoryID, newsID_subcategoryID, newsID_TitleWordID, newsID_AbstractWordID, newsID_titleEntityId_conf, newsID_abstractEntityId_conf).cuda()
        loss = model.loss(batch_predict, batch_label).cuda()
        summ.append(loss.data[0])

    average_loss = np.average(summ)
    return average_loss


def train(model, optimizer, loader, train_data, epoch):
    model.train()
    summ = []
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        print('Epoch: ', epoch+1, ' | Iteration: ', step + 1)
        processed_data = process_batch_data(train_data, batch_x.numpy())
        batch_user_history = processed_data[0]
        batch_user_short = processed_data[1]
        user_history_mask_selfattn = processed_data[2]
        user_history_mask_attn = processed_data[3]
        user_short_mask_selfattn = processed_data[4]
        user_short_mask_attn = processed_data[5]
        batch_user_valid = processed_data[6]
        batch_label = processed_data[7]

        # clear gradients
        optimizer.zero_grad()
        batch_predict = model(batch_user_history, batch_user_short, user_history_mask_selfattn, user_history_mask_attn, user_short_mask_selfattn, user_short_mask_attn, batch_user_valid, newsID_categoryID, newsID_subcategoryID, newsID_TitleWordID, newsID_AbstractWordID, newsID_titleEntityId_conf, newsID_abstractEntityId_conf).cuda()
        loss = model.loss(batch_predict, batch_label).cuda()
        print('Epoch ' + str(epoch+1) + ': ' + 'The ' + str(step + 1) + '/' + str(num_iterations) + '-th interation: loss: ' + str(loss.data[0]) + '\n')
        loss.backward()
        optimizer.step()
        summ.append(loss.data[0])

    average_loss = np.mean(summ)
    return average_loss


def train_and_evaluate(training_data, validation_data):
    history_encoder = TextEncoder(news_final_embed_size, history_num_head, history_attn_vector_size)
    recent_encoder = TextEncoder(news_final_embed_size, recent_num_head, recent_attn_vector_size)

    news_encoder = NewsEncoder(num_category, num_subcategory, title_embed_matrix, abstract_embed_matrix, entity_embed_matrix, num_head_text, num_head_entity, text_attn_vector_size, entity_attn_vector_size, news_final_attn_vector_size, news_final_embed_size)
    model = NRMS_new(history_encoder, recent_encoder, news_encoder, final_attn_vector_size)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=1e-3)

    best_vali_loss = 10000
    epoch = 0
    training_loss_epoch = []
    vali_loss_epoch = []

    train_data_index = torch.IntTensor(np.array(range(len(training_data))))
    train_data_index = Data.TensorDataset(data_tensor=train_data_index, target_tensor=train_data_index)
    train_loader = Data.DataLoader(dataset=train_data_index, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=False)

    validate_data_index = torch.IntTensor(np.array(range(len(validation_data))))
    validate_data_index = Data.TensorDataset(data_tensor=validate_data_index, target_tensor=validate_data_index)
    vali_loader = Data.DataLoader(dataset=validate_data_index, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

    for epoch in range(epochs):
        print('-----epoch ' + str(epoch+1) + '------')
        print('set batches')
        training_loss = train(model, optimizer, train_loader, training_data, epoch)
        training_loss_epoch.append(training_loss)
        print('The average loss of training set for the first ' + str(epoch) + ' epochs: ' + str(training_loss_epoch))

        evaluation_loss = evaluate(model, vali_loader, validation_data)
        vali_loss_epoch.append(evaluation_loss)
        print('The average loss of validation set for the first ' + str(epoch) + ' epochs: ' + str(vali_loss_epoch))

        if evaluation_loss < best_vali_loss:
            best_vali_loss = evaluation_loss
            torch.save(model, model_name)
        if epoch >= 5:
            "ealry stopping"
            near_loss = vali_loss_epoch[-5:]
            if near_loss == sorted(near_loss):  # loss increases for 5 consecutive epochs
                print("Best model found! Stop training, saving loss!")
                loss_train_vali = {'training loss': training_loss_epoch, 'testing loss': vali_loss_epoch}
                f = open(pack_loss, 'wb')
                pickle.dump(loss_train_vali, f)
                f.close()


if __name__ == '__main__':

    model_name = 'NRMS_new.pkl'
    pack_loss = 'NRMS_new_loss.pkl'

    print("loading hyper-parameters")
    args = parse_args()
    news_final_embed_size = args.news_final_embed_size
    history_num_head = args.history_num_head
    history_attn_vector_size = args.history_attn_vector_size
    recent_num_head = args.recent_num_head
    recent_attn_vector_size = args.recent_attn_vector_size
    batch_size = args.batch_size

    num_head_text = args.num_head_text
    num_head_entity = args.num_head_entity
    text_attn_vector_size = args.text_attn_vector_size
    entity_attn_vector_size = args.entity_attn_vector_size

    news_final_attn_vector_size = args.news_final_attn_vector_size
    final_attn_vector_size = args.final_attn_vector_size

    print("loading all dictionaries")

    f = open(args.category_id, 'rb')
    category_id = pickle.load(f)
    num_category = len(category_id['category_id'].keys())

    f = open(args.subcategory_id, 'rb')
    subcategory_id = pickle.load(f)
    num_subcategory = len(subcategory_id['subcategory_id'].keys())

    title_embed_matrix = np.load(args.TitleWordId_embeddings)
    abstract_embed_matrix = np.load(args.AbstractWordId_embeddings)
    entity_embed_matrix = np.load(args.EntityId_embeddings)

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

    print("loading training data")
    f = open(args.training_data, 'rb')
    training_data = pickle.load(f)['training_data_softmax']

    print("loading validation data")
    f = open(args.validation_data, 'rb')
    validation_data = pickle.load(f)['validation_data_softmax']

    # trim very long reading sequences
    for i, j in enumerate(training_data):
        if type(j[1]) != float and len(j[1]) > 60:
            training_data[i][1] = j[1][-60:]
        if type(j[2]) != float and len(j[2]) > 10:
            training_data[i][2] = j[2][-10:]

    for i, j in enumerate(validation_data):
        if type(j[1]) != float and len(j[1]) > 60:
            validation_data[i][1] = j[1][-60:]
        if type(j[2]) != float and len(j[2]) > 10:
            validation_data[i][2] = j[2][-10:]

    print("model training")
    training_data = np.array(training_data)
    validation_data = np.array(validation_data)
    f.close()
    training_len = len(training_data)
    num_iterations = training_len // batch_size
    print('number of training samples', training_len)
    train_and_validate(training_data, validation_data)

