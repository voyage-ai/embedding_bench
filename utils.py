import pickle
import json
import os
import argparse

class Config:
    """
    Configuration class for managing paths and settings in the project.
    Uses argparse to allow external configuration through command line.
    """
    def __init__(self, args):
        self.task_name = args.task_name
        self.embedding_models = args.embedding_models.split(',')
        self.topk = args.topk
        self.generative_model = args.generative_model
        self.data_path = f'data/{self.task_name}'
        self.meta_data_path = f'{self.data_path}/meta_data'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure paths and settings for the project.")
    parser.add_argument('--task-name', type=str, default='example_task', help='Task name for the dataset.')
    parser.add_argument('--embedding-models', type=str, default='voyage-large-2,text-embedding-3-large', help='Comma-separated list of embedding models.')
    parser.add_argument('--topk', type=int, default=3, help='Top K retrieved results.')
    parser.add_argument('--generative-model', type=str, default='gpt-4o-mini', help='Generative model name.')
    
    args = parser.parse_args()
    return args

def create_directories(config):
    """Create necessary directories for the project."""
    try:
        os.makedirs(config.meta_data_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        
def read_json_lines(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]