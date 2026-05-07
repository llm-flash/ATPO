# TreeHCA: Tree-Based Hindsight Credit Assignment for Multi-Turn LLM Agents

## Setup

### Local Retriever Tool Initialization

#### Environment
```
conda create -y -n retriever_env python=3.10
conda activate retriever_env
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

#### Download Retriever Data
```
mkdir rag_data
python rag_server/download.py
cat rag_data/part_* > rag_data/e5_Flat.index
gzip -d rag_data/wiki-18.jsonl.gz
```

### Training Environment Installation
```
conda create -n treehca_env python==3.10
conda activate treehca_env

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation

cd TreeHCA

pip install -r requirements.txt
```

### Dataset
```
mkdir rl_datasets
python data_process/hotpotqa_multihop_train.py
python data_process/multihop_test_merge.py
python data_process/nq_singlehop_train.py
python data_process/singlehop_test_merge.py
```

### Models
```
mkdir models
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

### Logs
```
mkdir logs
mkdir search_cache
```
