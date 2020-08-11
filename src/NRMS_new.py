import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.autograd import Variable
import pickle
from utils import *
import math


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.cpu()
    if mask is not None:
        scores = scores.masked_fill(Variable(mask) == 0, -1e9).cuda()

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()

        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return x


class AttnLayer(nn.Module):
    "made up of self-attn and attn"
    def __init__(self, self_attn_layer):
        super(AttnLayer, self).__init__()
        self.self_attn = self_attn_layer

    def forward(self, q, k, v, mask_selfattn, mask_attn, trans_weight_v, trans_weight_q):
        x = self.self_attn(q, k, v, mask_selfattn)
        scores = torch.matmul(torch.tanh(torch.matmul(x, trans_weight_v)), trans_weight_q)
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(Variable(mask_attn, requires_grad=False) == 0, -1e9)
        attend = F.softmax(scores, dim=1)

        check_all_zero_attn = torch.sum(mask_attn, dim=1)
        check_all_zero_attn = check_all_zero_attn.repeat(attend.size()[1], 1).transpose(0, 1)
        attend = attend.masked_fill(Variable(check_all_zero_attn, requires_grad=False) == 0, 0)

        embed = torch.matmul(x.transpose(1, 2), attend.unsqueeze(-1)).squeeze(-1)
        return embed


class TextEncoder(nn.Module):
    "including a self attention layer and a general attention layer"
    def __init__(self, word_embed_size, num_head, attn_vector_size):
        super(TextEncoder, self).__init__()
        self.self_attn_size = word_embed_size // num_head * num_head  #the size of an attention head * num_attn_head
        self.weight_query = nn.Parameter(torch.FloatTensor(word_embed_size, self.self_attn_size).cuda(), requires_grad=True)
        self.weight_key = nn.Parameter(torch.FloatTensor(word_embed_size, self.self_attn_size).cuda(), requires_grad=True)
        self.weight_value = nn.Parameter(torch.FloatTensor(word_embed_size, self.self_attn_size).cuda(), requires_grad=True)

        self.trans_weight_v = nn.Parameter(torch.FloatTensor(self.self_attn_size, attn_vector_size).cuda(), requires_grad=True)
        self.trans_weight_q = nn.Parameter(torch.FloatTensor(attn_vector_size, 1).cuda(), requires_grad=True)

        #c = copy.deepcopy
        attn = MultiHeadedAttention(num_head, word_embed_size)
        self.attention = AttnLayer(attn)

        self.parameters_init()

    def parameters_init(self):
        self.weight_query.data.uniform_(-1.0, 1.0)
        self.weight_key.data.uniform_(-1.0, 1.0)
        self.weight_value.data.uniform_(-1.0, 1.0)
        self.trans_weight_v.data.uniform_(-1.0, 1.0)
        self.trans_weight_q.data.uniform_(-1.0, 1.0)

    def forward(self, input_embeddings, mask_selfattn, mask_attn):
        "input_embeddings: 3D"
        #print(input_embeddings.size(), self.weight_query.size(), torch.sum(input_embeddings), torch.sum(self.weight_query))
        q = torch.matmul(input_embeddings, self.weight_query)
        k = torch.matmul(input_embeddings, self.weight_key)
        v = torch.matmul(input_embeddings, self.weight_value)
        attn = self.attention(q, k, v, mask_selfattn, mask_attn, self.trans_weight_v, self.trans_weight_q)
        return attn


class NewsEncoder(nn.Module):
    "Encode a piece of news, including topic, subtopic, title, abstract, title entities, abstract entities"
    def __init__(self, num_topic, num_subtopic, title_embed_matrix, abstract_embed_matrix, entity_embed_matrix, num_head_text, num_head_entity, text_attn_vector_size, entity_attn_vector_size, final_attn_vector_size, final_embed_size):
        super(NewsEncoder, self).__init__()
        self.num_topic = num_topic
        self.num_subtopic = num_subtopic
        self.title_embed_matrix = torch.FloatTensor(title_embed_matrix)
        self.abstract_embed_matrix = torch.FloatTensor(abstract_embed_matrix)
        self.entity_embed_matrix = torch.FloatTensor(entity_embed_matrix)
        self.self_attn_size_text = title_embed_matrix.shape[1] // num_head_text * num_head_text
        self.self_attn_size_entity = entity_embed_matrix.shape[1] // num_head_entity * num_head_entity

        self.final_embed_size = final_embed_size

        #attend topic, subtopic, title, abstract, title entity, abstract_entity...
        self.trans_weight_v = nn.Parameter(torch.FloatTensor(final_embed_size, final_attn_vector_size).cuda(), requires_grad=True)
        self.trans_weight_q = nn.Parameter(torch.FloatTensor(final_attn_vector_size, 1).cuda(), requires_grad=True)

        self.topic_embeddings = nn.Parameter(torch.FloatTensor(num_topic, final_embed_size).cuda(), requires_grad=True)
        self.subtopic_embeddings = nn.Parameter(torch.FloatTensor(num_subtopic, final_embed_size).cuda(), requires_grad=True)

        self.title_word_embeddings = nn.Embedding(title_embed_matrix.shape[0], title_embed_matrix.shape[1]).cuda()
        self.abstract_word_embeddings = nn.Embedding(abstract_embed_matrix.shape[0], abstract_embed_matrix.shape[1]).cuda()
        self.entity_ID_embeddings = nn.Embedding(entity_embed_matrix.shape[0], entity_embed_matrix.shape[1]).cuda()
        self.entity_ID_embeddings.weight.requires_grad = False

        self.title_words_linear = nn.Linear(self.self_attn_size_text, final_embed_size, bias=True).cuda()
        self.abstract_words_linear = nn.Linear(self.self_attn_size_text, final_embed_size, bias=True).cuda()
        self.title_entity_linear = nn.Linear(self.self_attn_size_entity, final_embed_size, bias=True).cuda()
        self.abstract_entity_linear = nn.Linear(self.self_attn_size_entity, final_embed_size, bias=True).cuda()

        self.title_encoder = TextEncoder(title_embed_matrix.shape[1], num_head_text, text_attn_vector_size)
        self.abstract_encoder = TextEncoder(abstract_embed_matrix.shape[1], num_head_text, text_attn_vector_size)
        self.title_entity_encoder = TextEncoder(entity_embed_matrix.shape[1], num_head_entity, entity_attn_vector_size)
        self.abstract_entity_encoder = TextEncoder(entity_embed_matrix.shape[1], num_head_entity, entity_attn_vector_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.topic_embeddings.data.uniform_(-1.0, 1.0)
        self.subtopic_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weight_v.data.uniform_(-1.0, 1.0)
        self.trans_weight_q.data.uniform_(-1.0, 1.0)
        self.title_word_embeddings.weight = nn.Parameter(torch.FloatTensor(self.title_embed_matrix).cuda())
        self.abstract_word_embeddings.weight = nn.Parameter(torch.FloatTensor(self.abstract_embed_matrix).cuda())
        self.entity_ID_embeddings.weight = nn.Parameter(torch.FloatTensor(self.entity_embed_matrix).cuda())

    def forward(self, batch_category, batch_subcategory, batch_title, batch_abstract, batch_title_entity, batch_abstract_entity, batch_title_mask_selfattn, batch_title_mask_attn, batch_abstract_mask_selfattn, batch_abstract_mask_attn, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn): # id
        "support batch calculation"

        category_embeddings = self.topic_embeddings[batch_category].unsqueeze(1)
        subcategory_embeddings = self.subtopic_embeddings[batch_subcategory].unsqueeze(1)
        title_embeddings = self.title_word_embeddings(Variable(torch.LongTensor(batch_title), requires_grad=False).cuda())
        abstract_embeddings = self.abstract_word_embeddings(Variable(torch.LongTensor(batch_abstract), requires_grad=False).cuda())

        batch_title_entity_id = batch_title_entity[:, :, 0]
        batch_title_entity_conf = batch_title_entity[:, :, 1]
        batch_title_embeddings = self.entity_ID_embeddings(Variable(torch.LongTensor(batch_title_entity_id), requires_grad=False).cuda())
        title_entity_embeddings = batch_title_embeddings * Variable(torch.FloatTensor(batch_title_entity_conf).unsqueeze(-1), requires_grad=False).cuda()

        batch_abstract_entity_id = batch_abstract_entity[:, :, 0]
        batch_abstract_entity_conf = batch_abstract_entity[:, :, 1]
        batch_abstract_embeddings = self.entity_ID_embeddings(Variable(torch.LongTensor(batch_abstract_entity_id), requires_grad=False).cuda())
        abstract_entity_embeddings = batch_abstract_embeddings * Variable(torch.FloatTensor(batch_abstract_entity_conf).unsqueeze(-1), requires_grad=False).cuda()

        # Feeding title information
        title_attn = self.title_encoder(title_embeddings, batch_title_mask_selfattn, batch_title_mask_attn)
        # Feeding abstract information
        abstract_attn = self.abstract_encoder(abstract_embeddings, batch_abstract_mask_selfattn, batch_abstract_mask_attn)
        # Feeding title entity information
        title_entity_attn = self.title_entity_encoder(title_entity_embeddings, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn)
        # Feeding abstract entity information
        abstract_entity_attn = self.abstract_entity_encoder(abstract_entity_embeddings, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn)

        title_attn = self.title_words_linear(title_attn).unsqueeze(1)
        abstract_attn = self.abstract_words_linear(abstract_attn).unsqueeze(1)
        title_entity_attn = self.title_entity_linear(title_entity_attn).unsqueeze(1)
        abstract_entity_attn = self.abstract_entity_linear(abstract_entity_attn).unsqueeze(1)

        concat = torch.cat((category_embeddings, subcategory_embeddings, title_attn, abstract_attn, title_entity_attn, abstract_entity_attn), 1)
        attend = F.softmax(torch.matmul(torch.tanh(torch.matmul(concat, self.trans_weight_v)), self.trans_weight_q),dim=1)
        attend = torch.matmul(concat.transpose(1, 2), attend).squeeze(-1)
        return attend


class NRMS_new(nn.Module):
    def __init__(self, history_encoder, recent_encoder, news_encoder, attn_vector_size):
        super(NRMS_new, self).__init__()
        self.history_encoder = history_encoder # an instance of TextEncoder
        self.recent_encoder = recent_encoder  # an instance of TextEncoder
        self.news_encoder = news_encoder  # an instance of NewsEncoder
        self.encode_behavior_size = history_encoder.self_attn_size

        self.trans_weight_v = nn.Parameter(torch.FloatTensor(self.encode_behavior_size, attn_vector_size).cuda())
        self.trans_weight_q = nn.Parameter(torch.FloatTensor(attn_vector_size, 1).cuda())
        self.parameters_init()

    def parameters_init(self):
        self.trans_weight_v.data.uniform_(-1.0, 1.0)
        self.trans_weight_q.data.uniform_(-1.0, 1.0)

    def forward(self, batch_user_history, batch_user_short, user_history_mask_selfattn, user_history_mask_attn, user_short_mask_selfattn, user_short_mask_attn, batch_user_valid, newsID_categoryID, newsID_subcategoryID, newsID_TitleWordID, newsID_AbstractWordID, newsID_titleEntityID_conf, newsID_abstractEntityID_conf):
        user_history = batch_user_history.reshape(1, -1).squeeze(0)
        batch_category = []
        batch_subcategory = []
        batch_title = []
        batch_abstract = []
        batch_title_entity = []
        batch_abstract_entity = []

        # padding words/entities for padded news / news with empty words/entities
        for i in user_history:
            if i in newsID_categoryID.keys():
                batch_category.append(newsID_categoryID[i])
                batch_subcategory.append(newsID_subcategoryID[i])

                if newsID_TitleWordID[i] == []:
                    batch_title.append([0])
                else:
                    batch_title.append(newsID_TitleWordID[i])

                if newsID_AbstractWordID[i] == []:
                    batch_abstract.append([0])
                else:
                    batch_abstract.append(newsID_AbstractWordID[i])

                if newsID_titleEntityID_conf[i] == []:
                    batch_title_entity.append([[0, 0]])
                else:
                    batch_title_entity.append(newsID_titleEntityID_conf[i])

                if newsID_abstractEntityID_conf[i] == []:
                    batch_abstract_entity.append([[0, 0]])
                else:
                    batch_abstract_entity.append(newsID_abstractEntityID_conf[i])
            else:  # mask
                batch_category.append(0)
                batch_subcategory.append(0)
                batch_title.append([0])
                batch_abstract.append([0])
                batch_title_entity.append([[0, 0]])
                batch_abstract_entity.append([[0, 0]])

        # padding and masking words

        pad_batch_title, batch_title_mask_selfattn, batch_title_mask_attn = self.pad_masking(batch_title)
        pad_batch_abstract, batch_abstract_mask_selfattn, batch_abstract_mask_attn = self.pad_masking(batch_abstract)

        pad_batch_title_entity, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn = self.pad_entity_masking(batch_title_entity)
        pad_batch_abstract_entity, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn = self.pad_entity_masking(batch_abstract_entity)

        history_news_embeddings = self.news_encoder(batch_category, batch_subcategory, pad_batch_title, pad_batch_abstract, pad_batch_title_entity, pad_batch_abstract_entity, batch_title_mask_selfattn, batch_title_mask_attn, batch_abstract_mask_selfattn, batch_abstract_mask_attn, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn)
        history_news_embeddings = history_news_embeddings.view(batch_user_history.shape[0], batch_user_history.shape[1], -1)  # 3D
        history_attention = self.history_encoder(history_news_embeddings, user_history_mask_selfattn, user_history_mask_attn)  # mask news

        user_short = batch_user_short.reshape(1, -1).squeeze(0)
        batch_category = []
        batch_subcategory = []
        batch_title = []
        batch_abstract = []
        batch_title_entity = []
        batch_abstract_entity = []

        for i in user_short:
            if i in newsID_categoryID.keys():
                batch_category.append(newsID_categoryID[i])
                batch_subcategory.append(newsID_subcategoryID[i])
                if newsID_TitleWordID[i] == []:
                    batch_title.append([0])
                else:
                    batch_title.append(newsID_TitleWordID[i])

                if newsID_AbstractWordID[i] == []:
                    batch_abstract.append([0])
                else:
                    batch_abstract.append(newsID_AbstractWordID[i])

                if newsID_titleEntityID_conf[i] == []:
                    batch_title_entity.append([[0, 0]])
                else:
                    batch_title_entity.append(newsID_titleEntityID_conf[i])

                if newsID_abstractEntityID_conf[i] == []:
                    batch_abstract_entity.append([[0, 0]])
                else:
                    batch_abstract_entity.append(newsID_abstractEntityID_conf[i])
            else:  # mask
                batch_category.append(0)
                batch_subcategory.append(0)
                batch_title.append([0])
                batch_abstract.append([0])
                batch_title_entity.append([[0, 0]])
                batch_abstract_entity.append([[0, 0]])
        # padding and masking words
        pad_batch_title, batch_title_mask_selfattn, batch_title_mask_attn = self.pad_masking(batch_title)
        pad_batch_abstract, batch_abstract_mask_selfattn, batch_abstract_mask_attn = self.pad_masking(batch_abstract)
        pad_batch_title_entity, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn = self.pad_entity_masking(batch_title_entity)
        pad_batch_abstract_entity, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn = self.pad_entity_masking(batch_abstract_entity)

        short_news_embeddings = self.news_encoder(batch_category, batch_subcategory, pad_batch_title, pad_batch_abstract, pad_batch_title_entity, pad_batch_abstract_entity, batch_title_mask_selfattn, batch_title_mask_attn, batch_abstract_mask_selfattn, batch_abstract_mask_attn, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn)
        short_news_embeddings = short_news_embeddings.view(batch_user_short.shape[0], batch_user_short.shape[1], -1)  # 3D
        short_attention = self.recent_encoder(short_news_embeddings, user_short_mask_selfattn, user_short_mask_attn)  # mask news

        concat = torch.cat((history_attention.unsqueeze(1), short_attention.unsqueeze(1)), 1)
        attend = F.softmax(torch.matmul(torch.tanh(torch.matmul(concat, self.trans_weight_v)), self.trans_weight_q), dim=1)
        attend = torch.matmul(concat.transpose(1, 2), attend).squeeze(-1)

        "User valid news embeddings"
        user_valid = batch_user_valid.reshape(1, -1).squeeze(0)
        batch_category = []
        batch_subcategory = []
        batch_title = []
        batch_abstract = []
        batch_title_entity = []
        batch_abstract_entity = []
        for i in user_valid:
            if i in newsID_categoryID.keys():
                batch_category.append(newsID_categoryID[i])
                batch_subcategory.append(newsID_subcategoryID[i])
                if newsID_TitleWordID[i] == []:
                    batch_title.append([0])
                else:
                    batch_title.append(newsID_TitleWordID[i])

                if newsID_AbstractWordID[i] == []:
                    batch_abstract.append([0])
                else:
                    batch_abstract.append(newsID_AbstractWordID[i])

                if newsID_titleEntityID_conf[i] == []:
                    batch_title_entity.append([[0, 0]])
                else:
                    batch_title_entity.append(newsID_titleEntityID_conf[i])

                if newsID_abstractEntityID_conf[i] == []:
                    batch_abstract_entity.append([[0, 0]])
                else:
                    batch_abstract_entity.append(newsID_abstractEntityID_conf[i])
            else:  # mask
                batch_category.append(0)
                batch_subcategory.append(0)
                batch_title.append([0])
                batch_abstract.append([0])
                batch_title_entity.append([[0, 0]])
                batch_abstract_entity.append([[0, 0]])
        # padding and masking words

        pad_batch_title, batch_title_mask_selfattn, batch_title_mask_attn = self.pad_masking(batch_title)
        pad_batch_abstract, batch_abstract_mask_selfattn, batch_abstract_mask_attn = self.pad_masking(batch_abstract)
        pad_batch_title_entity, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn = self.pad_entity_masking(batch_title_entity)
        pad_batch_abstract_entity, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn = self.pad_entity_masking(batch_abstract_entity)
        valid_news_embeddings = self.news_encoder(batch_category, batch_subcategory, pad_batch_title, pad_batch_abstract, pad_batch_title_entity, pad_batch_abstract_entity, batch_title_mask_selfattn, batch_title_mask_attn, batch_abstract_mask_selfattn, batch_abstract_mask_attn, batch_title_entity_mask_selfattn, batch_title_entity_mask_attn, batch_abstract_entity_mask_selfattn, batch_abstract_entity_mask_attn)
        #valid_news_embeddings = valid_news_embeddings.view(batch_user_valid.shape[0], -1)
        #dot = torch.sum(attend * valid_news_embeddings, dim=1)
        valid_news_embeddings = valid_news_embeddings.view(batch_user_valid.shape[0], -1, attend.size()[-1])
        dot = torch.sum(attend.unsqueeze(1) * valid_news_embeddings, dim=2)
        return dot

    def pad_masking(self, bat):  # pad all titles to the maximum length, then generate masks for attention
        batch = bat
        length = [len(i) for i in batch]
        max_len = max(length)
        for i in batch:
            i.extend([-1] * (max_len-len(i)))
        batch = np.array(list(batch), int)
        mask = batch.copy()
        mask[mask != -1] = 1
        mask[mask == -1] = 0
        mask_attn = mask.copy()
        mask1 = mask[:, :, np.newaxis]
        mask2 = mask[:, np.newaxis, :]
        mask = np.matmul(mask1, mask2)
        batch[batch == -1] = 0
        return batch, torch.IntTensor(mask), torch.IntTensor(mask_attn).cuda()

    def pad_entity_masking(self, bat):  # pad all entity to the maximum length, then generate masks for attention
        batch = bat
        length = [len(i) for i in batch]
        max_len = max(length)
        mask = np.zeros((len(batch), max_len))
        for i, j in enumerate(batch):
            mask[i][:len(j)] = 1
            j.extend([[0, 0]] * (max_len-len(j)))
        batch = np.array(list(batch))
        mask_attn = mask.copy()
        mask1 = mask[:, :, np.newaxis]
        mask2 = mask[:, np.newaxis, :]
        mask = np.matmul(mask1, mask2)
        return batch, torch.IntTensor(mask), torch.IntTensor(mask_attn).cuda()

    def loss(self, predict, label):
        label = Variable(torch.LongTensor(label), requires_grad=False).cuda()
        loss = nn.CrossEntropyLoss()
        output = loss(predict, label)
        return output





