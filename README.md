# NRMS_new-for-MIND

Source code of the NRMS_new model for the MIcrosoft News Dataset(MIND, https://msnews.github.io/) inspired by these papers:
1. NRMS-"Neural News Recommendation with Multi-Head Self-Attention" Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie (EMNLP 2019). 
2. MIND-"MIND: A Large-scale Dataset for News Recommendation" Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, Ming Zhou (ACL 2020)
3. LSTUR-"Neural News Recommendation with Long- and Short-term User Representations" Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu, Xing Xie (ACL 2019)
4. NAML-"Neural News Recommendation with Attentive Multi-View Learning" Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, Xing Xie (IJCAI 2019)

Brief model descriptions:

The whole model consists of three modules: TextEncoder, NewsEncoder, and NRMS_new, which is a hierarchical self-attention and additive attention structure to embed users' reading histories, each news in their reading histories, and each part of news at the same time. 
1. TextEncoder: same as the NewsEncoder in NRMS, a combindation of multi-head self-attention and additive attention mechanism to generate embeddings for a text, which can be title text, abstract text, title entities, and abstract entities. It also serves a natural framework to encode the reading history sequences of users to a single vector.
2. NewsEncoder: an additive attention that combines category embeddings, subcategory embeddings, title text embeddings, abstract text embeddings, title entity embeddings, and abstract entity embeddings and adds them up to a single vector for a piece of news.
3. NRMS_new: an additive attention that combines the embeddings of a user's reading histories and recent browsed news to a single vector and then performs dot product to the candidate news embeddings. Negative sampling is used in  model training. For each news browsed by a user (regarded as a positive sample), we randomly sample K news which are shown in the same impression but not clicked by the user (regarded as negative samples). We re-formulate the news click probability prediction problem as a pseudo (K + 1)-way classification task, and the lossfunction for model training is the negative log-likelihood of all positive samples.

Steps to run the code:
1. Set up two folders in the same directory: 'MINDlarge_train' for training data and 'MINDlarge_dev' validation data.
2. Data preprocess:
	python data_preprocess/behavior_preprocess.py
	python data_preprocess/news_preprocess.py
	(this two can be ran at the same time)
3. Model training:
	(1) Set up directories for files and hyperparameters in src/utils.py.
	(2) Model training: python src/main.py
3. Generate ranking list of news for test set:
	upcoming... 

