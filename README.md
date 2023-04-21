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


## 1. Download XMRec dataset from https://xmrec.github.io/
Category: Electronics
Data: ratings, metadata
Markets: Canada (ca), Germany (de), Spain (es), France (fr), India (in), Italy (it), Japan (jp), Mexico (mx), United Kingdom (uk), and United States (us)

## 2. Download Amazon reviews dataset from https://nijianmo.github.io/amazon/index.html
Category: Electronics
Data: metadata

## 3. Process data
```
python data_process.py
```

## 4. Pretrain ```us``` market
```
python pretrain.py
```

## 5. Fine-tune
We provide pretrained `CATSR-us-200.pth` to quick test fine-tune ```es``` market
```
python finetune.py
```
