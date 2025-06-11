import time
import torch
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from pipeline import RAGPipeline

class RADITTrainer:
    """
    Trains a RAGPipeline using Retrieval Augmented Dual Instruction Tuning
    """

    def __init__(self, pipeline: RAGPipeline, train_dataset: Dataset, test_dataset: Dataset | None = None, epochs=1, batch_size=100):
        self.pipeline = pipeline
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size

    def radit_lm(self, batch: dict[str, list]) -> float:
        losses = []
        for _, data_point in tqdm(pd.DataFrame(batch).iterrows(), '\t\tModel Fine-Tuning'):

            query = data_point["query"]
            golden_answer = data_point["answer"]
            documents = data_point["documents"]

            retrieved_documents, _ = self.pipeline.retrieve_documents(query, documents)
            generator_loss = self.pipeline.calculate_generator_losses(query, golden_answer, retrieved_documents).mean()

            self.pipeline.model_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            self.pipeline.model_optimizer.step()
            losses += [generator_loss.item()]
        return sum(losses) / len(losses)

    def radit_encoder(self, batch: dict[str, list]) -> float:
        losses = []
        for _, data_point in tqdm(pd.DataFrame(batch).iterrows(), '\t\tEncoder Fine-Tuning'):

            query = data_point["query"]
            golden_answer = data_point["answer"]
            documents = data_point["documents"]

            encoder_loss = self.pipeline.calculate_encoder_loss(query, golden_answer, documents)

            self.pipeline.encoder_optimizer.zero_grad(set_to_none=True)
            encoder_loss.backward()
            self.pipeline.encoder_optimizer.step()
            losses += [encoder_loss.item()]

        return sum(losses) / len(losses)

    def radit(self, batch: dict[str, list]) -> tuple[float, float]:
        if self.pipeline.cuda:
            torch.cuda.empty_cache()
        generator_loss = self.radit_lm(batch)
        encoder_loss = self.radit_encoder(batch)
        return generator_loss, encoder_loss
    
    def evaluate_pipeline(self) -> float | None:
        if self.test_dataset is None:
            return None

        losses = []
        for data_point in tqdm(self.test_dataset):

            query = data_point["query"]
            golden_answer = data_point["answer"]
            documents = data_point["documents"]

            with torch.no_grad():
                retrieved_documents, _ = self.pipeline.retrieve_documents(query, documents)
                generator_loss = self.pipeline.calculate_generator_losses(query, golden_answer, retrieved_documents).mean()
                losses += [generator_loss.item()]
        return pd.Series(losses).mean()

    
    def train(self, output_filepath: str) -> list[dict]:
        self.pipeline.encoder_model.train()
        self.pipeline.model.train()
        losses = [{
            'generator': [],
            'encoder': [],
            'evaluation': None
        } for _ in range(self.epochs)]
        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"Starting epoch {epoch + 1}")
            count = 0
            for batch in self.train_dataset.iter(batch_size=self.batch_size):
                count += 1
                batch_start = time.time()

                generator_loss, encoder_loss = self.radit(batch)

                batch_end = time.time()
                print(f"\tBatch {count} took {batch_end - batch_start:.0f} seconds.")
                print(f"\tgenerator loss: {generator_loss:.4f}, encoder loss: {encoder_loss:.4f}")
                print(f"\tEpoch age: {(batch_end - epoch_start)/60:.0f} minutes.")

                losses[epoch]['generator'] += [generator_loss]
                losses[epoch]['encoder'] += [encoder_loss]
                with open(output_filepath, 'w') as file:
                    json.dump(losses, file, indent=4)
            
            losses[epoch]['evaluation'] = self.evaluate_pipeline()
            with open(output_filepath, 'w') as file:
                json.dump(losses, file, indent=4)
                
        return losses