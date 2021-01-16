from torch.utils.data import Dataset,DataLoader        
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer,BertForMultipleChoice
import torch
import time
import random
import math
class IR_Dataset(Dataset):
    def __init__(self,tokenizer,query_pos_doc_tuple_list,query_neg_name_dict,documents_df,query_df,neg_doc_num):
        self.tokenizer = tokenizer
        self.query_pos_doc_tuple_list = query_pos_doc_tuple_list
        self.query_neg_name_dict = query_neg_name_dict
        self.documents_df = documents_df
        self.query_df = query_df
        self.neg_doc_num = neg_doc_num
        
    def __len__(self):
        return len(self.query_pos_doc_tuple_list)
    
    def __getitem__(self,index):
        query_id,pos_doc_name = self.query_pos_doc_tuple_list[index]
        query_context = self.query_df[self.query_df['query_id'] == query_id]['query_text'].item()
        
        pos_doc = self.documents_df[self.documents_df['doc_id'] == pos_doc_name]['doc_text'].item()
        input_doc_list = [pos_doc]
        input_label_list = [1]
        
        neg_doc_name_list = self.query_neg_name_dict[query_id]
        for loop in range(self.neg_doc_num):
            neg_doc_index = np.random.randint(0,len(neg_doc_name_list))
            neg_doc_name = neg_doc_name_list[neg_doc_index]
            neg_doc = self.documents_df[self.documents_df['doc_id'] == neg_doc_name]['doc_text'].item()
            input_doc_list.append(neg_doc)
            input_label_list.append(0)

        input_list = list(zip(input_doc_list,input_label_list))
        random.shuffle(input_list)
        input_doc_list,input_label_list = zip(*input_list)
        for index,label in enumerate(input_label_list):
            if label == 1:
                pos_doc_index = index
        
        target = torch.tensor(pos_doc_index)
        encode_dict = self.tokenizer([query_context]*len(input_doc_list),input_doc_list,padding = 'max_length',truncation = True,return_tensors = 'pt')
        return encode_dict['input_ids'],encode_dict['token_type_ids'],encode_dict['attention_mask'],target
    
class Test_Dataset(Dataset):
    def __init__(self,tokenizer,query_BMdoc_list,documents_df,test_queries_df):
        self.tokenizer = tokenizer
        self.query_BMdoc_list = query_BMdoc_list
        self.test_queries_df = test_queries_df
        self.documents_df = documents_df
        
    def __len__(self):
        return len(self.query_BMdoc_list)
        
    def __getitem__(self,index):
        query_id,document_name,bm_score = self.query_BMdoc_list[index]
        query = self.test_queries_df[self.test_queries_df['query_id'] == query_id]['query_text'].item()
        document = self.documents_df[self.documents_df['doc_id'] == document_name]['doc_text'].item()
        
        encode_dict = tokenizer([query],[document],padding = 'max_length',truncation = True,return_tensors = 'pt')
        return encode_dict['input_ids'],encode_dict['token_type_ids'],encode_dict['attention_mask'],bm_score,query_id,document_name
    
def predict(model,test_loader,alpha):
    submit_df = pd.DataFrame(columns = ['query_id','ranked_doc_ids','score'])
    previous_query_id = ''
    submit_df_index = 0
    score_dict = {}
    
    for data_index,data in enumerate(test_loader):
        input_ids = data[0].to(device)
        token_type_ids = data[1].to(device)
        attention_mask = data[2].to(device)
        bm_scores = data[3]
        query_ids = data[4]
        document_names = data[5]
        
        bert_outputs = model(input_ids = input_ids,token_type_ids = token_type_ids,attention_mask = attention_mask)
        bert_score = bert_outputs[0]
                             
        for index in range(len(bert_score)):
            query_id = (str)(query_ids[index].numpy())
            if query_id != previous_query_id:
                if score_dict != {}:
                    score_tuple_list = sorted(score_dict.items(), key=lambda x:x[1])
                    score_tuple_list.reverse()
                    ranked_doc = ''
                    doc_score = ''
                    for doc_id,score in score_tuple_list:
                        ranked_doc += doc_id + ' '
                        doc_score += (str)(score) + ' '
                    submit_df.loc[submit_df_index] = [previous_query_id,ranked_doc,doc_score]
                    submit_df_index += 1
                score_dict = {}
            
            score = bm_scores[index] + alpha * bert_score[index]
            doc_name = ''.join(document_names[index])
            score_dict[doc_name] = (float)(score)
            previous_query_id = query_id
        if data_index % 500 == 0:
            print(f"finish data:{data_index} / {len(test_loader)},time:{time.strftime('%X')}")
    
    score_tuple_list = sorted(score_dict.items(), key=lambda x:x[1])
    score_tuple_list.reverse()
    ranked_doc = ''
    doc_score = ''
    for doc_id,score in score_tuple_list:
        ranked_doc += doc_id + ' '
        doc_score += (str)(score) + ' '
    submit_df.loc[submit_df_index] = [query_id, ranked_doc, doc_score]
           
    return submit_df
    
def get_df(folder,df_name):
    file = pd.read_csv(os.path.join(folder,df_name))
    return pd.DataFrame(file)

def not_nan(pos_doc_name,documents_df):
    doc_content = documents_df[documents_df['doc_id'] == pos_doc_name]['doc_text'].item()
    if doc_content != 'nan':
        return True
    else:
        return False

def initialize_training_dataset(train_queries_df,documents_df):
    query_neg_name_dict = {}
    query_pos_tuple_list = []
    for loop,query_info in train_queries_df.iterrows():
        query_id = query_info['query_id']
        pos_doc_names = query_info['pos_doc_ids'].split()
        for pos_doc_name in pos_doc_names:
            if not_nan(pos_doc_name,documents_df):
                query_pos_tuple_list.append((query_id,pos_doc_name))
        
        
        neg_doc_names_row = query_info['bm25_top1000'].split()
        neg_doc_names = []
        for neg_doc_name in neg_doc_names_row:
            if neg_doc_name not in pos_doc_names:
                neg_doc_names.append(neg_doc_name)

        query_neg_name_dict[query_id] = neg_doc_names
    return query_neg_name_dict, query_pos_tuple_list
    
def initialize_testing_dataset(test_queries_df):
    query_BMdoc_list = []
    for loop, query_info in test_queries_df.iterrows():
        query_id = query_info['query_id']
        bm_score_list = query_info['bm25_top1000_scores'].split()
        document_list = query_info['bm25_top1000'].split()
        for document, bm_score in zip(document_list, bm_score_list):
            query_BMdoc_list.append((query_id, document, (float)(bm_score)))
    return query_BMdoc_list

def Bert_train(dataloader,optimizer,model,epochs,batch_size):
    for epoch in range(epochs):
        running_loss = 0
        for index,data in enumerate(dataloader):
            input_ids,token_type_ids,attention_mask,label = [d.to(device) for d in data]
                
            optimizer.zero_grad()
            output = model(input_ids = input_ids,
                           token_type_ids = token_type_ids,
                            attention_mask = attention_mask,
                            labels = label)
            loss = output[0]
            running_loss += loss.item()
        
            loss.backward()
            optimizer.step()
            if index % 500 == 0:
                print(f"train index:{index} / {len(dataloader)} time:{time.strftime('%X')}")
        print(f"loss:{running_loss},time:{time.strftime('%X')}")

def main():
    print(f"start at {time.strftime('%X')}")
    #open file
    # -------------------------------------------------------------------------
    folder = '../input/ntust-ir2020-homework6/'
    documents_df = get_df(folder, 'documents.csv')
    train_queries_df = get_df(folder, 'train_queries.csv')
    test_queries_df = get_df(folder, 'test_queries.csv')

    #set tokenizer and running device
    # -------------------------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")

    #set training parameter
    # -------------------------------------------------------------------------
    training_batch_size = 4
    training_lr = 3e-5
    epoch = 1
    neg_doc_num = 3

    #create training dataloader 
    # -------------------------------------------------------------------------
    query_neg_name_dict, query_pos_tuple_list = initialize_training_dataset(train_queries_df, documents_df)
    train_dataset = IR_Dataset(tokenizer,query_pos_tuple_list,query_neg_name_dict,documents_df,train_queries_df,neg_doc_num)
    train_loader = DataLoader(train_dataset,batch_size = training_batch_size,shuffle = True)

    #initialize model and train
    # -------------------------------------------------------------------------
    model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
    optimizer = torch.optim.AdamW(model.parameters(),lr = training_lr)
    model = model.to(device)
    model.train()
    print(f"train start at {time.strftime('%X')}")
    Bert_train(train_loader,optimizer,model,epoch,training_batch_size)
    torch.cuda.empty_cache()
    print(f"train end at {time.strftime('%X')}")

    #save model
    # -------------------------------------------------------------------------
    save_name = 'bert_1229'
    torch.save(model.state_dict(), save_name)

    #create test dataloader
    # -------------------------------------------------------------------------
    query_BMdoc_list = initialize_testing_dataset(test_queries_df)
    test_dataset = Test_Dataset(tokenizer,query_BMdoc_list,documents_df,test_queries_df)
    test_loader = DataLoader(test_dataset,batch_size = 8)

    #predict answer
    # -------------------------------------------------------------------------
    print(f"predict start at {time.strftime('%X')}")
    submit_df = predict(model,test_loader,1.85)
    print(f"predict end at {time.strftime('%X')}")

    submit_df.to_csv('answer.csv',index = False)

if __name__ == "__main__":
    main()     