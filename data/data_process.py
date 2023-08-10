import gzip
import json
import os

import torch
import pandas as pd
import numpy as np

def parse(path_tmp):
    g = gzip.open(path_tmp, "rb")
    for line in g:
        yield json.loads(line)
        
def getDF(path_tmp, format):
    if format == "json":
        i = 0
        df = {}
        for d in parse(path_tmp):
            df[i] = d
            i += 1
        df = pd.DataFrame.from_dict(df, orient="index")
        df = df[["asin", "title", "description"]]
        return df
    elif format == "txt":
        df = pd.read_csv(path_tmp, compression="gzip", header=None, sep=" ", names=["reviewerID", "asin", "ratings", "date"] )
        return df
    
def list2str(input):
    if len(input) == 0:
        return ""
    else:
        return input[0]
    
def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    
def generate_item_emb(df, emboutputpath):
    print("Using BERT to generate item embeddings...")
    all_sentences = df["text_feature"].tolist()
    from transformers import AutoModel, AutoTokenizer
    device = set_device('0')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    
    embeddings = []
    start, batch_size = 0, 4
    while start < len(all_sentences):
        sentences = all_sentences[start: start + batch_size]
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                    truncation=True, return_tensors='pt').to(device)
        outputs = model(**encoded_sentences)
        cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
        embeddings.append(cls_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)
    #Sentences are encoded by calling model.encode()
    asin_list = df["asin"].tolist()
    temp = pd.DataFrame({"item_id:token": asin_list, "item_emb:float_seq": embeddings.tolist()})
    temp['item_emb:float_seq'] = temp['item_emb:float_seq'].apply(lambda x: ' '.join(map(str, x)))
    temp.to_csv(emboutputpath, index=False)
    print(f"Done! text embeddings have been saved into: {emboutputpath}")
    return temp

if __name__ == "__main__":
    # load amazon meta data to extract item's text feature
    print("Loading Amazon meta data...")
    # meta_df = getDF('Amazon/metadata/meta_Electronics.json.gz', 'json')
    
    # select the items that are in the market
    # itemset = set()
    # print("Selecting items that are in the market...")
    market_list = ['ca', 'de', 'fr', 'in', 'jp', 'mx', 'uk', 'us']
    # for m in market_list:
    #     df = pd.read_csv(m+'_5core.txt', sep=' ')
    #     itemset = itemset.union(set(df.itemId.unique()))
        
    # selected_meta_df = meta_df[meta_df['asin'].isin(itemset)]
    # selected_meta_df.drop_duplicates(subset=['asin'], inplace=True)
    
    # assert len(selected_meta_df) == len(itemset)
    
    # # generate item's text feature
    # selected_meta_df["description"] = selected_meta_df.description.apply(list2str)
    # selected_meta_df["text_feature"] = selected_meta_df[["title", "description"]].agg(" ".join, axis=1)

    # selected_meta_df = selected_meta_df.drop(columns=["title", "description"])
    # selected_meta_df = selected_meta_df.reset_index(drop=True)
    
    # generate item's text embedding
    # emb_df = generate_item_emb(selected_meta_df, "item_emb.csv")
    emb_df = pd.read_csv("item_emb.csv")
    
    # generate recbole dataset
    if not os.path.exists('../dataset'):
        os.mkdir('../dataset')
    for m in market_list:
        print(f"Generating {m} dataset...")
        df = pd.read_csv(m+'_5core.txt', sep=' ')
        itemset = set(df.itemId.unique())
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df['timestamp'] = df.date.values.astype(np.int64) // 10 ** 9
        df = df[['userId', 'itemId', 'timestamp']].rename(columns={'userId': 'user_id:token', 'itemId': 'item_id:token', 'timestamp': 'timestamp:float'})
        df = df[['user_id:token', 'item_id:token', 'timestamp:float']]
        if not os.path.exists(f'../dataset/{m}'):
            os.mkdir(f'../dataset/{m}')
        df.to_csv(f'../dataset/{m}/{m}.inter', sep='\t', index=False)
        m_emb_df = emb_df[emb_df['item_id:token'].isin(itemset)]
        m_emb_df.to_csv(f'../dataset/{m}/{m}.item', sep='\t', index=False)