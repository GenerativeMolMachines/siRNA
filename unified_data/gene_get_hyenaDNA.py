import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import tqdm
import json
from transformers import AutoConfig
from typing import Optional
import torch.nn as nn
from huggingface_hub import snapshot_download
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

df = pd.read_csv('E:\\My_projects\\gene_embeddings.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

access_token = "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"  # Replace with your actual token

with torch.no_grad():
    # Load HyenaDNA model
    model = AutoModelForCausalLM.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True, token=access_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True, token=access_token)  # Use same tokenizer as model
    model.eval() # Set model to evaluation mode


    results = []
    unique_sequences = df['gene_sequence'].unique()

    for sequence in tqdm.tqdm(unique_sequences):
        # Tokenize the sequence
        enc = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True).to(device)

        # Get embeddings from the last hidden state
        outputs = model(**enc, output_hidden_states=True)  # Pass tokenized input through model
        hidden_states = outputs.hidden_states[-1]  # Access the last hidden state

        emb = torch.mean(hidden_states.cpu()[0], dim=0)  # Average over sequence length


        results.append({
            "input": sequence,
            "embedding": emb.tolist()
        })

with open('gene_hyenadna.json', "w") as file:
    json.dump(list(results), file)
