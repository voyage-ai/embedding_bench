# Head-to-Head Comparison of Embedding Models

This repository offers a straightforward and cost-effective method to compare the retrieval quality of various embedding models using a collection of unpaired queries and documents. For each embedding model, the process involves leveraging a language model to score the top $k$ documents retrieved and then calculating the average score across these documents and queries. This allows users to perform a head-to-head comparison by analyzing the average scores.

**Detailed Steps**:

*Document Embedding and Storage*:
- Embed all documents using the embedding model.
- Insert the embeddings into a vector database.

*Query Processing*:
- For each query:
    - Generate the query embedding.
    - Retrieve the top $k$ nearest documents based on the embedding.

*Document Scoring*:
- For each pair of (query, retrieved document):
    - Use a language model to evaluate if the document contains useful information for answering the query.

*Score Calculation*:
- Calculate the average score for the top $k$ documents for each query.

We repeat this process for each embedding model and output the overall average score. Users can perform a head-to-head comparison by analyzing the average scores.

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