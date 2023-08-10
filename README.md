# CAT-SR
In repo is for RecSys 2023 paper.

## Requirements

```
recbole==1.1.1
python==3.8.5
cudatoolkit==11.3.1
pytorch==1.12.1
pandas==1.3.0
transformers==4.18.0
```


### 1. Copy preprocessed XMRec dataset from [FOREC](https://github.com/hamedrab/FOREC/tree/main/DATA/proc_data) or [MA](https://github.com/samarthbhargav/efficient-xmrec/tree/main/DATA2/proc_data)
put dataset into ```data``` file.

### 2. Download Amazon meta dataset from https://nijianmo.github.io/amazon/index.html
Category: Electronics
Data: metadata
put dataset into ```data/Amazon/metadata/meta_Electronics.json.gz``` directory

### 3. Process data
```
cd data
python data_process.py
```

### 4. Pretrain ```us``` market
```
python pretrain.py
```

### 5. Fine-tune
Take finetune Canada(ca) as an example
```
python finetune.py --dataset ca
```
