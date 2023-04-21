import numpy as np
import json
import pandas as pd
import gc
import gzip
import os

from tqdm import tqdm


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
    
def generate_k_core(df, k):
    # on users
    # tmp1 = df.groupby(["reviewerID"], as_index=False)["asin"].count()
    # tmp1.rename(columns={"asin": "cnt_item"}, inplace=True)
    # on items
    tmp2 = df.groupby(["asin"], as_index=False)["reviewerID"].count()
    tmp2.rename(columns={"reviewerID": "cnt_user"}, inplace=True)

    df = df.merge(tmp2, on=["asin"])
    query = "cnt_user >= %d" % k
    df = df.query(query).reset_index(drop=True).copy()
    df.drop(["cnt_user"], axis=1, inplace=True)
    del tmp2
    gc.collect()
    return df

def info(data):
    """ num of user, item, max/min uid/itemID, total interaction"""
    user = set(data["reviewerID"].tolist())
    item = set(data["asin"].tolist())
    print("number of user: ", len(user))
    print("max user ID: ", max(user))
    print("Min user ID: ", min(user))
    print("number of Item: ", len(item))
    print("Interactions: ", len(data))
    
def user_limit(df, k=5):
    # select users with more than (include) k interacted items
    tmp1 = df.groupby(["reviewerID"], as_index=False)["asin"].count()
    tmp1.rename(columns={"asin": "cnt_item"}, inplace=True)
    df = df.merge(tmp1, on=["reviewerID"])
    query = "cnt_item >= %d" % k
    df = df.query(query).reset_index(drop=True).copy()
    df.drop(["cnt_item"], axis=1, inplace=True)
    del tmp1
    gc.collect()
    return df

def str2int(in_str):
    """
    input sample: "[11506, 10463, 34296, 15541]"
    """
    data_list = in_str.strip()[1:-1].split(",")  # remove "[]" and split
    data = list(map(int, data_list))
    return data

def clean_rating_data(df):
    # pre-processing on the original dataset TODO: change parameters accordingly.
    print("==========Information of Original dataset==========")
    info(df)
    # clean with overall verified value. And selected time span.
    df = df[df.ratings >= 3]                 # select positive interactions.

    # # k-core
    # df = generate_k_core(df, k=5)  # item
    # print("======================Information of %d-core dataset:" % 5)
    # info(df)

    df = user_limit(df, k=5)   # user
    print("==========Information of dataset where all users have at least %d interaction==========" % 5)
    info(df)
    return df

def clean_meta_data(df):
    print("==========Information of Original dataset==========")
    print(f"Len of samples: {len(df)}")
    df["title_len"] = df.title.apply(len)
    # only keeps the item that has a valid title
    df = df[df.title_len > 0]
    print("==========Information of dataset with valid titles==========")
    print(f"Len of samples: {len(df)}")
    return df

def sort_sequence(df):
    # Convert User_ID and Item_ID and sort by reviewerID and time
    # df["reviewerID"] = pd.Categorical(df.reviewerID).codes
    # df["reviewerID"] = df["reviewerID"] + 1
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df['timestamp'] = df.date.values.astype(np.int64) // 10 ** 9
    df.sort_values(by=["reviewerID", "timestamp"], inplace=True)
    df = df[['reviewerID', 'asin', 'timestamp']].rename(columns={'reviewerID': 'user_id:token', 'asin': 'item_id:token', 'timestamp': 'timestamp:float'})
    return df
    
def extract_text_features(city_list):
    file_path = os.path.join(AMAZONROOT, f"meta_{CATEGORY}.json.gz")
    print(f"Starting extract {city_list} text features through {file_path}...")
    city_set_dict = dict()
    df = getDF(file_path, "json")
    # get asin set
    asin_set = set(df.asin.unique().tolist())

    for city in city_list:
        temp_df = getDF(os.path.join(XMRECROOT, f"{city}", f"metadata_{city}_{CATEGORY}.json.gz"), "json")
        temp_set = set(temp_df.asin.unique().tolist())
        city_set_dict[city] = temp_set
    
    if len(city_set_dict.keys()) == 1:
        city_set = city_set_dict[city_list[0]]
    else:
        city_set = city_set_dict[city_list[0]]
        for city in city_list[1:]:
            city_set.update(city_set_dict[city])

    common_asin_set = asin_set & city_set

    df_filtered = df[df.asin.isin(common_asin_set)]
    df_filtered.loc[:, "title_len"] = df_filtered.title.str.len()
    df_filtered = df_filtered.sort_values("title_len")
    df_filtered = df_filtered.drop_duplicates(subset="asin", keep="last")
    def list2str(input):
        if len(input) == 0:
            return ""
        else:
            return input[0]
    df_filtered["description"] = df_filtered.description.apply(list2str)
    df_filtered["text_feature"] = df_filtered[["title", "description"]].agg(" ".join, axis=1)
    df_filtered = df_filtered.drop(columns=["title", "description", "title_len"])
    df_filtered = df_filtered.reset_index()
    df_filtered["idx"] = df_filtered.index
    df_filtered.to_csv(TEXTOUTPUTPATH, index=False)
    print(f"Done! text fuature dataset has been stored in: {TEXTOUTPUTPATH}")
    return df_filtered

def generate_item_emb(df):
    print(f"generating item's text embedding...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    #Our sentences we like to encode
    sentences = df["text_feature"].tolist()
    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    asin_list = df["asin"].tolist()
    temp = pd.DataFrame({"item_id:token": asin_list, "item_emb:float_seq": embeddings.tolist()})
    temp['item_emb:float_seq'] = temp['item_emb:float_seq'].apply(lambda x: ' '.join(map(str, x)))
    temp.to_csv(EMBOUTPUTPATH, index=False)
    print(f"Done! text embeddings have been saved into: {EMBOUTPUTPATH}")
    return temp
    
def generate_atomic_files(domain, merged_emb_df):
    #--------------------Stage One: Load rating dataset & meta dataset
    print("\n----->Stage One: Load rating and meta dataset<-----\n")
    form = "txt"
    prefix = "ratings"
    ratings_df = getDF(os.path.join(XMRECROOT, f"{domain}", f"{prefix}_{domain}_{CATEGORY}.{form}.gz", ), form)
    
    form = "json"
    prefix = "metadata"
    meta_df = getDF(os.path.join(XMRECROOT, f"{domain}", f"{prefix}_{domain}_{CATEGORY}.{form}.gz", ), form)
    
    #--------------------Stage Two: Clean ratings dataset and meta dataset
    # For ratings dataset, we clean data dynamicly, to keep the source domain has sufficient users
    print("\n----->Stage Two: Clean ratings dataset<-----\n")
    # remove item with invalid title (title len equals 0)
    meta_df = clean_meta_data(meta_df)
    # remove invalid item in rating dataset
    ratings_df = ratings_df.join(meta_df.set_index("asin"), on="asin").drop(columns=["title", "title_len"])
    # keeps k core for rating dataset
    ratings_df = clean_rating_data(ratings_df)
    # sort ratings dataset
    ratings_df = sort_sequence(ratings_df)
    # check whether the domain directory is exist
    domain_path = os.path.join(OUTPUTROOT, f"{domain}")
    if not os.path.exists(domain_path):
        os.mkdir(domain_path)
    ratings_df.to_csv(os.path.join(domain_path, f"{domain}.inter"), index=False, sep='\t')
    print("\n----->Stage Three: generating atomic files<-----\n")
    # filter embeddings according to the asin in the specific market 
    item_set = set(ratings_df["item_id:token"].unique().tolist())
    selected_embeddings = merged_emb_df[merged_emb_df["item_id:token"].isin(item_set)]
    # save embeddings for the market
    selected_embeddings.to_csv(os.path.join(domain_path, f"{domain}.item"), index=False, sep='\t')
    print(f"Saved {domain}_embeddings")
    print("Done!")
    
if __name__ == "__main__":
    """Processing XMRec dataset, which could extract item text features(English) for each market(country).
    The whole process contain extracting English format text features through Amazon metadata then generating
    text features embeddings and mapping them into each market for XMRec dataset. The outputs are follows Recbole Atomic file format.
    Args:
        city_list: the market list, e.g. city_list = ["us", "ca", "jp", "mx"]
        CATEGORY: the data category. Amazon and XMRec have the same category name.
        AMAZONROOT: the root path of Amazon meta dataset
        XMRECROOT: the root path of XMRec dataset
        OUTPUTROOT: the root path of atomic fiels
        TEXTOUTPUTPATH: the path of whole text features for each item. csv format.
        EMBOUTPUTPATH: the path of whole embeddings for each item. csv format.
        
    Returns:
        Atomic files
    """ 
    city_list = ['ca', 'de', 'es', 'fr', 'in', 'it', 'jp', 'mx', 'uk', 'us']
    CATEGORY = "Electronics"
    AMAZONROOT = f"Amazon/metadata" # Amazon meta dataset root path 
    XMRECROOT = "XMRec/raw_data" # XMRec dataset root path
    OUTPUTROOT = f"dataset" # output dir
    TEXTOUTPUTPATH = os.path.join(OUTPUTROOT, f"meta_{CATEGORY}_{''.join(map(str, city_list))}.csv")
    EMBOUTPUTPATH = os.path.join(OUTPUTROOT, f"emb_{CATEGORY}_{''.join(map(str, city_list))}.csv")
    
    # retrive all item's text features through Amazon meta dataset in English format
    # generate text feature embeddings for each item
    if os.path.exists(EMBOUTPUTPATH):
        df = pd.read_csv(EMBOUTPUTPATH)
    elif os.path.exists(TEXTOUTPUTPATH):
        df = pd.read_csv(TEXTOUTPUTPATH)
        df = generate_item_emb(df)
    else:
        df = extract_text_features(city_list)
        df = generate_item_emb(df)

    for city in city_list:
        generate_atomic_files(city, df)
    
    