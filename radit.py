import time
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from pipeline import RAGPipeline

class RADITTrainer:
    """
    Trains a RAGPipeline using Retrieval Augmented Dual Instruction Tuning
    """

    def __init__(self, pipeline: RAGPipeline, dataset: Dataset, epochs=1, batch_size=100):
        self.pipeline = pipeline
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size

    def radit_lm(self, batch: dict[str, list]) -> None:
        for _, data_point in tqdm(pd.DataFrame(batch).iterrows(), '\t\tModel Fine-Tuning'):

            query = data_point["query"]
            golden_answer = data_point["answer"]
            documents = data_point["documents"]

            encoder_loss = self.pipeline.calculate_encoder_loss(query, golden_answer, documents)

            self.pipeline.encoder_optimizer.zero_grad(set_to_none=True)
            encoder_loss.backward()
            self.pipeline.encoder_optimizer.step()

    def radit_encoder(self, batch: dict[str, list]) -> None:
        for _, data_point in tqdm(pd.DataFrame(batch).iterrows(), '\t\tEncoder Fine-Tuning'):

            query = data_point["query"]
            golden_answer = data_point["answer"]
            documents = data_point["documents"]

            encoder_loss = self.pipeline.calculate_encoder_loss(query, golden_answer, documents)

            self.pipeline.encoder_optimizer.zero_grad(set_to_none=True)
            encoder_loss.backward()
            self.pipeline.encoder_optimizer.step()

    def radit(self, batch: dict[str, list]) -> None:
        if self.pipeline.cuda:
            torch.cuda.empty_cache()
        self.radit_lm(batch)
        self.radit_encoder(batch)
    
    def train(self):
        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"Starting epoch {epoch + 1}")
            count = 0
            for batch in self.dataset.iter(batch_size=self.batch_size):
                count += 1
                batch_start = time.time()

                self.radit(batch)

                batch_end = time.time()
                print(f"\tBatch {count} took {batch_end - batch_start:.0f} seconds.")
                print(f"\tEpoch age: {batch_end - epoch_start:.0f} seconds.")