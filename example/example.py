from glob import glob
from os.path import exists, join, basename
from tqdm import tqdm
import json
from matplotlib import pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from quadtreed3 import Quadtree, Node
from scipy.sparse import csr_matrix
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from typing import Tuple
from io import BytesIO
from umap import UMAP

import pandas as pd
import numpy as np
import ndjson  
import requests
import urllib
import wizmap
from sentence_transformers import SentenceTransformer

SEED = 20230501

plt.rcParams['figure.dpi'] = 300


# Load dataset

import os
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
   
with open(r'/home/mxma/project/llm-attacks/experiments/results/individual_behavior_controls.json','r',encoding = 'utf-8') as f:
    data = json.load(f)

data_prompt = data['controls']

print('Loaded', len(data_prompt), 'prompts.')

# # Load the pre-trained embedding model

# model = SentenceTransformer('all-MiniLM-L6-v2').quantize(8).half().cuda()

# # Encode all 25k reviews
# BATCH_SIZE = 128
# prompt_embeddings = model.encode(data_prompt, batch_size=BATCH_SIZE, show_progress_bar=True)



from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


model_path = '/home/mxma/DIR/all-MiniLM-L6-v2'
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
BATCH_SIZE = 128
# Tokenize sentences
encoded_input = tokenizer(data_prompt, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
prompt_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
prompt_embeddings = F.normalize(prompt_embeddings, p=2, dim=1)






print(f'Embedding shape: {prompt_embeddings.shape}')

reducer = UMAP(metric='cosine')
embeddings_2d = reducer.fit_transform(prompt_embeddings)


plt.title(f'UMAP Projected Embeddings of {len(data_prompt)} ')
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=0.1, alpha=0.2)
plt.show()


xs = embeddings_2d[:, 0].astype(float).tolist()
ys = embeddings_2d[:, 1].astype(float).tolist()
texts = data_prompt

data_list = wizmap.generate_data_list(xs, ys, texts)
grid_dict = wizmap.generate_grid_dict(xs, ys, texts)


# Save the JSON files
wizmap.save_json_files(data_list, grid_dict, output_dir='./')