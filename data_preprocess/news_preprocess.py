import pandas as pd
import numpy as np
import pickle
import random
from ast import literal_eval


def save_category_subcategory(category, subcategory, newsid):
    cate_all = category.drop_duplicates()
    subcate_all = subcategory.drop_duplicates()
    id_category = dict(zip(range(len(cate_all)), cate_all))
    category_id = dict(zip(cate_all, range(len(cate_all))))
    category_dict = {'id_category': id_category, 'category_id': category_id}
    id_subcate = dict(zip(range(len(subcate_all)), subcate_all))
    subcate_id = dict(zip(subcate_all, range(len(subcate_all))))
    subcate_dict = {'id_subcategory': id_subcate, 'subcategory_id': subcate_id}
    f = open('category_id.pkl', 'wb')
    pickle.dump(category_dict, f)
    f = open('subcategory_id.pkl', 'wb')
    pickle.dump(subcate_dict, f)
    f.close()
    print(category_dict, subcate_dict)

    cateid = category.apply(lambda x: category_id[x])
    subcateid = subcategory.apply(lambda x: subcate_id[x])
    newsid_cateid = dict(zip(newsid, cateid))
    newsid_subcateid = dict(zip(newsid, subcateid))
    f = open('newsID_categoryID.pkl', 'wb')
    pickle.dump(newsid_cateid, f)
    f = open('newsID_subcategoryID.pkl', 'wb')
    pickle.dump(newsid_subcateid, f)
    f.close()


def save_word_embeddings(column_words, newsid, file_name):
    "save word embeddings for title or abstract"
    column_word = column_words.apply(lambda x: str(x))
    split = column_word.str.split(' ')
    split = split.apply(lambda x: [i.rstrip('\'s') for i in x])
    split = split.apply(lambda x: [i.rstrip(',.!?\'\"') for i in x])
    split = split.apply(lambda x: [i.lstrip(',.!?\'\"') for i in x])
    words = split.tolist()
    word_vocab = list(set([word for sublist in words for word in sublist]))

    "save word embeddings from glove"
    f = open('../glove/glove_dict.pkl', 'rb')
    glove = pickle.load(f)
    glove_words = glove.keys()
    word_vocab = [i for i in word_vocab if i in glove_words]
    word_embeddings = np.array([glove[i] for i in word_vocab])
    print(len(word_embeddings))
    np.save(file_name+'Id_embeddings.npy', word_embeddings)
    print('word embeddings matrix generated')

    id_word = dict(zip(range(len(word_vocab)), word_vocab))
    word_id = dict(zip(word_vocab, range(len(word_vocab))))
    word_dict = {'id_word': id_word, 'word_id': word_id}
    f = open(file_name+'_id.pkl', 'wb')
    pickle.dump(word_dict, f)
    print('word dictionary generated')

    split = split.apply(lambda x: [word_id[i] for i in x if i in word_vocab])
    newsid_wordsids = dict(zip(newsid, split))
    f = open('newsID_'+file_name+'ID.pkl', 'wb')
    pickle.dump(newsid_wordsids, f)
    f.close()
    print('newsID_wordIDs generated')


def save_entity(newsdata):
    entity1 = pd.read_csv('../MINDlarge_train/entity_embedding.vec', header=None, sep='\t')
    entity2 = pd.read_csv('../MINDlarge_dev/entity_embedding.vec', header=None, sep='\t')
    entity = pd.concat((entity1, entity2), axis=0)
    entity = entity.drop_duplicates()
    entity_IDs = entity[0].drop_duplicates()
    entityID_id = dict(zip(entity_IDs, range(len(entity_IDs))))
    id_entityID = dict(zip(range(len(entity_IDs)), entity_IDs))
    entity_dict = {'entityID_id': entityID_id, 'id_entityID': id_entityID}
    f = open('entityID_id.pkl', 'wb')
    pickle.dump(entity_dict, f)
    entity['vector'] = entity.iloc[:, 1:101].values.tolist()
    entity['id'] = entity[0].apply(lambda x: entityID_id[x])
    entity = entity.sort_values(by=['id'])
    entity_embeddings = np.array(list(entity['vector']))
    np.save('EntityId_embeddings.npy', entity_embeddings)

    newsdata = newsdata.copy()
    newsdata['TitleEntities'] = newsdata['TitleEntities'].apply(lambda x: literal_eval(x) if pd.notnull(x) else [])
    newsdata_titleentity = newsdata.loc[newsdata['TitleEntities'].str.len()!=0]
    newsdata_titleentity['TitleEntity_Confidence'] = newsdata_titleentity['TitleEntities'].apply(lambda x: [[entityID_id[i['WikidataId']], float(i['Confidence'])] for i in x if i['WikidataId'] in entityID_id.keys()])
    newsdata_titleno = newsdata.loc[newsdata['TitleEntities'].str.len()==0]
    newsdata_titleno['TitleEntity_Confidence'] = newsdata_titleno['TitleEntities']
    newsdata_all = pd.concat((newsdata_titleentity.loc[:, ['NewsID', 'TitleEntity_Confidence']], newsdata_titleno.loc[:, ['NewsID', 'TitleEntity_Confidence']]), axis=0)
    news_title_entity = dict(zip(newsdata_all['NewsID'], newsdata_all['TitleEntity_Confidence']))
    print(len(news_title_entity))
    f = open('newsID_titleEntityId_conf.pkl', 'wb')
    pickle.dump(news_title_entity, f)

    newsdata = newsdata.copy()
    newsdata['AbstractEntities'] = newsdata['AbstractEntities'].apply(lambda x: literal_eval(x) if pd.notnull(x) else [])
    newsdata_absentity = newsdata.loc[newsdata['AbstractEntities'].str.len() != 0]
    newsdata_absentity['AbstractEntity_Confidence'] = newsdata_absentity['AbstractEntities'].apply(lambda x: [[entityID_id[i['WikidataId']], float(i['Confidence'])] for i in x if i['WikidataId'] in entityID_id.keys()])
    newsdata_absentityno = newsdata.loc[newsdata['AbstractEntities'].str.len() == 0]
    newsdata_absentityno['AbstractEntity_Confidence'] = newsdata_absentityno['AbstractEntities']
    newsdata_all = pd.concat((newsdata_absentity.loc[:, ['NewsID', 'AbstractEntity_Confidence']], newsdata_absentityno.loc[:, ['NewsID', 'AbstractEntity_Confidence']]), axis=0)
    news_abstract_entity = dict(zip(newsdata_all['NewsID'], newsdata_all['AbstractEntity_Confidence']))
    print(len(news_abstract_entity))
    f = open('newsID_abstractEntityId_conf.pkl', 'wb')
    pickle.dump(news_abstract_entity, f)
    f.close()


if __name__ == '__main__':
    news_train = pd.read_csv('../MINDlarge_train/news.tsv', sep='\t', header=None)
    news_train.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
    news_dev = pd.read_csv('../MINDlarge_dev/news.tsv', sep='\t', header=None)
    news_dev.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
    data = pd.concat((news_train, news_dev), axis=0)
    data = data.drop_duplicates()

    save_category_subcategory(data['Category'], data['SubCategory'], data['NewsID'])

    save_word_embeddings(data['Title'], data['NewsID'], 'TitleWord')
    save_word_embeddings(data['Abstract'], data['NewsID'], 'AbstractWord')
    save_entity(data)
