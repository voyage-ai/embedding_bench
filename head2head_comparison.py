import json
import re
import dotenv
from tqdm import tqdm
from utils import parse_arguments, Config, read_json_lines
import sys

# Load environment variables
dotenv.load_dotenv()
from generation import get_gpt_results

def load_prompt_from_file(prompt_file_path):
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()
    return prompt

def label_pair(query, doc, prompt, generative_model):
    input_text = f'Given a document and a query:\nQuery:\n{query}\nDocument:\n{doc}\n{prompt}'
    response_text = get_gpt_results(input_text, generative_model)
    score = parse_digit(response_text)
    return score

def parse_digit(response_text):
    pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(pattern, response_text)
    if match:
        numbers = re.findall(r'\d+', match.group(1))
        if numbers:
            return int(numbers[0])
    return None

if __name__ == "__main__":
    args = parse_arguments()
    config = Config(args)
    corpus = {item['id']: item['text'] for item in read_json_lines(f'{config.data_path}/corpus.jsonl')}
    queries = {item['id']: item['text'] for item in read_json_lines(f'{config.data_path}/queries.jsonl')}
    
    prompt = load_prompt_from_file(f'./default_prompt.txt')

    for embedding_model_name in config.embedding_models:
        collection_name = embedding_model_name.replace('-', '_')
        with open(f'{config.meta_data_path}/{collection_name}.json', 'r') as f:
            retrieved_dict = json.load(f)
        score_dict = {}
        for qid in tqdm(retrieved_dict.keys(), desc="Queries", leave=False):
            score_dict[qid] = {}
            for cid in retrieved_dict[qid]:
                score_dict[qid][cid] = label_pair(queries[qid], corpus[cid], prompt, config.generative_model)
        with open(f'{config.meta_data_path}/{collection_name}_scores.json', 'w') as f:
            json.dump(score_dict, f, indent=4)
        
        # Calculate average score for each query
        average_scores = {}
        for qid, scores in score_dict.items():
            total_score = sum(scores.values())
            average_score = total_score / len(scores)
            average_scores[qid] = average_score

        # Calculate overall average score
        overall_average_score = sum(average_scores.values()) / len(average_scores)

        # Print overall average score and flush the output buffer
        print(f"Overall average score for {embedding_model_name}: {overall_average_score}\n")