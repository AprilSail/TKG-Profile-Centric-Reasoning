# TKG-Profile-Centric-Reasoning
Official code for the paper **Open-World Temporal Knowledge Graph Reasoning: A Profile-Centric Evaluation Benchmark**.  

The current version of the code fully supports result reproduction. More detailed files (e.g., embedding model downloads, entity/relation embedding generation, and test result files for additional baseline models) will be released soon.

## 1. Dataset Construction
In our paper, we construct profile-centric evaluation versions for six publicly available TKG datasets: ICEWS14, ICEWS18, ICEWS05-15, SAIKG, YAGO, and WIKI in `raw_datasets`.

### 1.1 Detailed Information of Backbone Datasets
The key statistics of these datasets are summarized in the table below:
| Dataset      | Entities | Relations | Time  | Train    | Valid   | Test    | Granularity | Dataset Link                              |
|--------------|----------|-----------|-------|----------|---------|---------|-------------|-------------------------------------------|
| ICEWS05-15   | 10,488   | 251       | 4,017 | 368,868  | 46,302  | 46,159  | 24 hours    | https://dataverse.harvard.edu/dataverse/icews |
| ICEWS14      | 7,128    | 230       | 365   | 74,845   | 8,514   | 7,371   | 24 hours    | https://dataverse.harvard.edu/dataverse/icews |
| ICEWS18      | 23,033   | 256       | 304   | 373,018  | 45,995  | 49,545  | 24 hours    | https://dataverse.harvard.edu/dataverse/icews |
| SAIKG        | 5,758    | 38        | 363   | 177,220  | 14,045  | 14,045  | 1.5 years   | https://github.com/AprilSail/SAIKG        |
| WIKI         | 12,554   | 24        | 232   | 539,286  | 67,538  | 63,110  | 1 year      | https://en.wikipedia.org/wiki/Data_set    |
| YAGO         | 10,623   | 10        | 189   | 161,540  | 19,523  | 20,026  | 1 year      | https://yago-knowledge.org                |

### 1.2 Profile-Centric Datasets
We provide constructed Profile-Centric Evaluation datasets in directory format:
- The folder `datasets/Profile_Centric_Dataset_Q` corresponds to different Q settings in PrH@Q-K (Q ∈ {3,4,5,6,7,8,9,10}).
- Each folder contains test sets for datasets including `ICEWS05-15-PR`, `ICEWS14-PR`, `ICEWS18-PR`, `SAIKG-PR`, `WIKI-PR`, and `YAGO-PR` (used for baseline model evaluation).

### 1.3 Build Your Own Profile-Centric Dataset
To construct a profile-centric evaluation version for your custom TKG dataset, follow these steps:

#### Step 1: Unseen Entity Visualization (Optional)
Visualize the distribution of unseen entities in your dataset:
```bash
python dataset_construct/unseen_entity_visualization.py
```

#### Step 2: Generate Entity/Relation Embeddings
Generate and save embeddings for entities and relations using pre-trained embedding models:
```bash
python dataset_construct/entity_relation_embedding_generation.py
```

#### Step 3: Build Profile-Centric Evaluation Dataset
Generate the profile-centric dataset using the entity/relation embeddings:
```bash
python dataset_construct/profile_centric_dataset_construction.py
```
Note: Training and validation sets for the TKG datasets can be generated using the same steps above.

## 2. Baseline Model Evaluation
In our paper, we retrained and re-evaluated **18 TKG reasoning methods across 5 categories**. To facilitate the application of our evaluation paradigm to new baseline models in future research, we select one representative method from each category (TTransE, RE-GCN, CyGNet, TANGO-DistMult, TimeTraveler) and provide their retrained prediction results as evaluation examples. The prediction results of these 5 models are stored in the `baselines` folder.

### 2.1 Baseline Model Categories
The 18 baseline models are categorized as follows:
1. **Score function models**: TTransE, TATransE, TADistMult;
2. **Graph modelling models**: RE-Net, RE-GCN, HIP Network, TiRGN, LogCL, DiMNet;
3. **Time embedding models**: xERTE, CyGNet;
4. **Time aware models**: TANGO-DistMult, TANGO-TuckER, CEN, CENET, Co-CyGNet, Co-CENET;
5. **Path modelling models**: TimeTraveler.

### 2.2 Detailed Information of Baseline Models
| Method                  | Technique         | Publication | Year | Code Link                                      |
|-------------------------|-------------------|-------------|------|------------------------------------------------|
| TTransE                 | Score Function    | WWW         | 2018 | https://github.com/mklimasz/TransE-PyTorch     |
| TATransE                | Score Function    | EMNLP       | 2018 | https://github.com/bsantraigi/TA_TransE        |
| TADistmult              | Score Function    | EMNLP       | 2018 | https://github.com/pyg-team/pytorch_geometric  |
| RE-Net                  | Graph Modelling   | EMNLP       | 2020 | https://github.com/INK-USC/RE-Net              |
| xERTE                   | Time Embedding    | ICLR        | 2020 | https://github.com/TemporalKGTeam/xERTE        |
| CyGNet                  | Time Embedding    | AAAI        | 2021 | https://github.com/CunchaoZ/CyGNet             |
| RE-GCN                  | Graph Modelling   | SIGIR       | 2021 | https://github.com/Lee-zix/RE-GCN              |
| TANGO-DistMult/TuckER   | Time Aware        | EMNLP       | 2021 | https://github.com/TemporalKGTeam/TANGO        |
| TimeTraveler            | Path Modelling    | EMNLP       | 2021 | https://github.com/JHL-HUST/TITer              |
| HIP Network             | Graph Modelling   | IJCAI       | 2021 | https://github.com/Yongquan-He/HIP-network     |
| CEN                     | Time Aware        | ACL         | 2022 | https://github.com/Lee-zix/CEN                 |
| TiRGN                   | Graph Modelling   | IJCAI       | 2022 | https://github.com/Liyyy2122/TiRGN             |
| CENET                   | Time Aware        | AAAI        | 2023 | https://github.com/xyjigsaw/CENET              |
| LogCL                   | Graph Modelling   | ICDE        | 2024 | https://github.com/WeiChen3690/LogCL           |
| Co-CyGNet/CENET         | Time Aware        | INF.SCI     | 2025 | https://github.com/AprilSail/SAIKG             |
| DiMNet                  | Graph Modelling   | ACL         | 2025 | https://github.com/hhdo/DiMNet                 |

### 2.3 Evaluation Pipeline
#### Step 1: Save Model Prediction Results
Generate prediction results of TKGR baseline models on the original dataset file `test.txt`, save the top-K predictions (corresponding to K in PrH@Q-K) in dictionary format:
```bash
python baseline_evaluate/model_query_prediction.py
```
#### Step 2: Evaluate Profile-Centric Reasoning Performance
Evaluate the reasoning performance of models on the Profile-Centric-Reasoning task based on their prediction results:
```bash
python baseline_evaluate/model_evaluation.py
```
