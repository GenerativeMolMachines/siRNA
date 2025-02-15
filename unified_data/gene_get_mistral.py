import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
from sys import argv




df = pd.read_csv('E:\\My_projects\\gene_embeddings.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

access_token = "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"

with torch.no_grad():
    model = AutoModel.from_pretrained('mistralai/Mistral-7B-v0.1', torch_dtype=torch.float16, token=access_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', token=access_token)
    results = []
    unique_sequences = df['gene_sequence'].unique()

    for reaction in tqdm.tqdm(unique_sequences):

        enc = tokenizer(reaction, return_tensors="pt", truncation=True).to(device)
        emb = model(**enc).last_hidden_state.cpu()[0][-1]
        results.append({
               "input": reaction,
               "embedding": emb.tolist()
       })

with open('gene_mistral.json', "w") as file:
   json.dump(
       list(results),
       file)