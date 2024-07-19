# Head-to-Head Comparison of Embedding Models

This repository provides a simple and cost-efficient method for comparing the retrieval quality of several embedding models using a collection of unpaired queries and documents. It leverages GPT to score the top $k$ documents retrieved by different models and outputs the average score.

## Installation

We recommend using Conda for the installation.

```bash
conda create -n embedding_bench_env python=3.10
conda activate embedding_bench_env
pip install -r requirements.txt
```

## Prepare data
Store the queries and documents in the `data/{task_name}` directory:

`queries.jsonl` should include the query id and query text.
```json
"00000": {"text": "This is the text of the first query."},
"00001": {"text": "This is the text of the second query."},
```

`corpus.jsonl` should contain the document id and document text.
```json
"000000000": {"text": "This is the text of the first document."},
"000000001": {"text": "This is the text of the second document."},
```

## API Key Configuration

We need to invoke APIs for embedding models and generative language models. To configure the API keys using environment variables, please store them in `.env` file located in the root directory of your project.

## Embed and Retrieve

For each query, use a set of embedding models to get its top $k$ candidate documents. Embeddings are saved in ./data/{task_name}/embedding.db. Generated candidates are saved under the folder `./data/{task_name}/meta_data` in the format of `{query_id: [document_ids]}`.

```bash
python embed_retrieve.py --task-name example_task --embedding-models voyage-large-2,text-embedding-3-large --topk 3
```

## Head to head comparison using GPT

For each retrieved document, use GPT to determine if they constitute a relevant match. The document and query are assessed as a pair based on criteria divided into four levels: reject (label 1), borderline reject (label 2), borderline accept (label 3), and accept (label 4). The final scores are saved in `./data/{task_name}/meta_data/{embedding_model_name}_score.jsonl` in the format of `{query_id: {document_id: score}}`.

```bash
python head2head_comparison.py --task-name example_task --topk 3 --generative-model gpt-4o
```