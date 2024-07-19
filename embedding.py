import time
import logging
import openai
import tiktoken
import voyageai
from abc import ABC
import os

def get_embedding_model(model_name):
    if model_name == "text-embedding-3-large":
        return OpenAIEmbeddingModel(model_name, os.environ.get("OPENAI_API_KEY"))
    elif model_name == "voyage-large-2":
        return VoyageEmbeddingModel(model_name, os.environ.get("VOYAGE_API_KEY"))
    else:
        raise ValueError(f"Invalid model name: {model_name}")

class EmbeddingModel(ABC):

    def forward(self, batch):
        raise NotImplementedError


class OpenAIEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name, api_key):
        
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_length = 8191
        self.dim = 3072

    def forward(self, texts):
        
        tokens = [self.tokenizer.encode(text, disallowed_special=()) for text in texts]
        if self.max_length:
            tokens = [t[:self.max_length] for t in tokens]
        while True:
            try:
                result = self.client.embeddings.create(
                    input=tokens,
                    model=self.model_name,
                )
                embeddings = [d.embedding for d in result.data]
                return embeddings
            except openai.RateLimitError as e:
                logging.error(e)
                time.sleep(30)
            except openai.InternalServerError as e:
                logging.error(e)
                time.sleep(60)


class VoyageEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name, api_key):

        super().__init__()
        self.client = voyageai.Client(
            api_key=api_key,
            max_retries=3,
        )
        self.model_name = model_name
        self.max_retries = 5
        self.dim = 1536

    def forward(self, texts):

        num_retries = 0
        while num_retries < self.max_retries:
            try:
                result = self.client.embed(
                    texts, model=self.model_name, truncation=True)
                return result.embeddings
            except voyageai.error.RateLimitError as e:
                logging.error(e)
                time.sleep(30)
            except voyageai.error.ServiceUnavailableError as e:
                logging.error(e)
                time.sleep(60)
            num_retries += 1
        raise RuntimeError(
            f"Voyage embedding API did not respond in {num_retries} retries."
        )
        