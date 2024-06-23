import pandas as pd
import numpy as np

import faiss
import transformers
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings('ignore')

PATH_TO_EMBEDS = 'rag_files/compressed_array.npz'
PATH_TO_DF = 'rag_files/compressed_dataframe.csv.gz'

embeddings = np.load(PATH_TO_EMBEDS)
df_data = pd.read_csv(PATH_TO_DF, compression='gzip')

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embeddings['array_data']

embed_length = embeddings.shape[1]

num_centroids = 5

quantizer = faiss.IndexFlatL2(embed_length)

index = faiss.IndexIVFFlat(quantizer, embed_length, num_centroids)

index.train(embeddings)

if not index.is_trained:
    raise ValueError("error happenned in the training")
else:
    print("training done")

index.add(embeddings)
index.nprobe = 5

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")

pipeline=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=100,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
    )

def summary_generator(text):
    llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
    summary_prompt = "summarize this text: " + text
    return llm(summary_prompt)

def run_faiss_search(query_text, top_k):

    query = [query_text]

    query_embedding = model.encode(query)


    scores, index_vals = quantizer.search(query_embedding, top_k)
    
    index_vals_list = index_vals[0]
    
    return index_vals_list
    

def run_rerank(index_vals_list, query_text):
    
    chunk_list = list(df_data['prepared_text'])

    pred_strings_list = [chunk_list[item] for item in index_vals_list]

    cross_input_list = []

    for item in pred_strings_list:

        new_list = [query_text, item]

        cross_input_list.append(new_list)


    df = pd.DataFrame(cross_input_list, columns=['query_text', 'pred_text'])

    df['original_index'] = index_vals_list

    cross_scores = cross_encoder.predict(cross_input_list)

    df['cross_scores'] = cross_scores

    df_sorted = df.sort_values(by='cross_scores', ascending=False)
    
    df_sorted = df_sorted.reset_index(drop=True)

    pred_list = []

    for i in range(0,len(df_sorted)):

        text = df_sorted.loc[i, 'pred_text']

        original_index = df_sorted.loc[i, 'original_index']
        arxiv_id = df_data.loc[original_index, 'id']
        cat_text = df_data.loc[original_index, 'cat_text']
        title = df_data.loc[original_index, 'title']

        link_to_pdf = f'https://arxiv.org/pdf/{arxiv_id}'

        item = {
            'arxiv_id': arxiv_id,
            'link_to_pdf': link_to_pdf,
            'cat_text': cat_text,
            'title': title,
            'abstract': text
        }

        pred_list.append(item)

    return pred_list


def jsonify_search_results(pred_list, num_results_to_print):
    
    results = {}
    for i in range(0,num_results_to_print):
        
        pred_dict = pred_list[i]
        
        link_to_pdf = pred_dict['link_to_pdf']
        abstract = summary_generator(pred_dict['abstract'])
        cat_text = pred_dict['cat_text']
        title = pred_dict['title']

        results[f"Result {i+1}"] = {
            'link_to_pdf': link_to_pdf,
            'abstract': abstract,
            'cat_text': cat_text,
            'title': title
        }

    return results

        
    

def run_arxiv_search(query_text, num_results_to_print, top_k=300):
    
    pred_index_list = run_faiss_search(query_text, top_k)

    pred_list = run_rerank(pred_index_list, query_text)

    return jsonify_search_results(pred_list, num_results_to_print)