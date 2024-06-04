# CAT-SR

This is the official PyTorch implementation for the [paper]():
> Chen Wang, Ziwei Fan, Liangwei Yang, Mingdai Yang, Xiaolong Liu, Zhiwei Liu, Philip Yu. Pre-Training with Transferable Attention for Addressing Market
Shifts in Cross-Market Sequential Recommendation. KDD 2024.

---

## Overview

In this study, we introduce the **C**ross-market **A**ttention **T**ransferring with **S**equential **R**ecommendation (**CAT-SR**) framework, tailored specifically for cross-market recommendation (CMR) scenarios. CMR poses unique challenges such as strict privacy regulations that limit data sharing, lack of user overlap, and consistent item sets across different international markets. These aspects are further compounded by market-specific variations in user preferences and item popularity, known as market shifts.

To effectively address these hurdles and enhance recommendation accuracy across disparate markets, CATSR employs a sophisticated approach that leverages a preconditioning strategy focusing on item-item correlations and incorporates an innovative selective self-attention mechanism. This mechanism facilitates the transfer of focused learning across markets. Additionally, the framework enhances adaptability through the integration of query and key adapters, which are designed to capture and adjust to market-specific nuances in user behavior.

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
Put data file into ```data``` directory. For example: ```data/ca_5core.txt```

### 2. Download [Amazon meta dataset](https://nijianmo.github.io/amazon/index.html)
Category: Electronics

Data: metadata

Put dataset into ```data/Amazon/metadata``` directory. For example ```data/Amazon/metadata/meta_Electronics.json.gz```

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
