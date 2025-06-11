import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity, kl_div
from torch.nn import CrossEntropyLoss

class RAGPipeline:
    """
    A simple RAG pipeline for efficient training.
    """

    def __init__(self, model, tokenizer, encoder_model, k=1, cuda=False, lr=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.encoder_model = encoder_model
        self.cuda = cuda
        self.model_optimizer = Adam(self.model.parameters(), lr=lr)
        self.encoder_optimizer = Adam(self.encoder_model.parameters(), lr=lr)

    def retrieve_documents(self, query: str, documents: list[str]):
        if len(documents) < self.k:
            raise ValueError("Invalid input: number of candidate documents is smaller than retrieval 'k'.")

        tokenized_inputs = self.encoder_model.tokenizer([query] + documents, padding=True, return_tensors="pt")
        if self.cuda:
            for key in tokenized_inputs.keys():
                tokenized_inputs[key] = tokenized_inputs[key].to('cuda')

        outputs = self.encoder_model(tokenized_inputs)
        embedded_query = outputs['sentence_embedding'][0]
        embedded_documents = outputs['sentence_embedding'][1:]

        scores = cosine_similarity(embedded_query, embedded_documents)

        top_k_indices = torch.topk(scores, self.k).indices
        retrieved_docs = [documents[i] for i in top_k_indices]
        retriever_scores = scores[top_k_indices]

        return retrieved_docs, retriever_scores

    def build_prompt(self, query: str, retrieved_documents: list[str]):
        background = "\n".join(retrieved_documents)
        prompt = (
            f"You are tasked to respond to a query. "
            f"You will also be given some background that might be relevant to the query.\n"
            f"BACKGROUND:"
            f"=========================================\n"
            f"{background}"
            f"=========================================\n"
            f"QUERY:"
            f"=========================================\n"
            f"{query}"
            f"=========================================\n"
            f"YOUR ANSWER:\n"
        )
        return prompt

    def calculate_generator_losses(self, query: str, golden_answer: str, retrieved_documents: list[str]):

        augmented_prompts = [self.build_prompt(query, [document]) for document in retrieved_documents]

        tokenized_prompts = self.tokenizer(text=augmented_prompts, padding=False)
        tokenized_golden_answer = self.tokenizer(text=golden_answer, padding=False)

        # Save the state of the warning, then disable it
        warning_state = self.tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        try:
            inputs = self.tokenizer.pad(
                encoded_inputs={
                    'input_ids': [
                        tokenized_prompt + tokenized_golden_answer['input_ids'] for tokenized_prompt in tokenized_prompts['input_ids']
                    ]
                },
                padding=True,
                return_tensors="pt"
            )
            labels = self.tokenizer.pad(
                encoded_inputs={
                    'input_ids': [
                        [-100] * len(tokenized_prompt) + tokenized_golden_answer['input_ids'] for tokenized_prompt in tokenized_prompts['input_ids']
                    ]
                },
                padding=True,
                return_tensors="pt"
            )
        finally:
            # Restore the state of the warning.
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

        labels['input_ids'][labels['attention_mask'] == 0] = -100
        inputs['labels'] = labels['input_ids']

        if self.cuda:
            # Move all tensors to GPU
            inputs['input_ids'] = inputs['input_ids'].to('cuda')
            inputs['labels'] = inputs['labels'].to('cuda')
            inputs['attention_mask'] = inputs['attention_mask'].to('cuda')

        outputs = self.model(**inputs)

        shifted_logits = outputs['logits'][:, :-1, :]
        shifted_labels = inputs['input_ids'][:, 1:]

        loss_function = CrossEntropyLoss(reduction="none", ignore_index=-100)
        losses_per_label = loss_function(
            torch.transpose(shifted_logits, -2, -1),
            shifted_labels
        )

        shifted_mask = (shifted_labels != -100)
        losses = torch.sum(losses_per_label * shifted_mask, dim=1) / torch.sum(shifted_mask, dim=1)

        return losses

    def calculate_generator_scores(self, query: str, golden_answer: str, retrieved_documents: list[str], temperature: float = 0.01):
        losses = self.calculate_generator_losses(query, golden_answer, retrieved_documents)
        # Normalize to get p(c|x,y)
        scores = torch.softmax(-losses / temperature, dim=-1)
        return scores

    def calculate_encoder_loss(self, query: str, golden_answer: str, documents: list[str]):
        retrieved_documents, retriever_scores = self.retrieve_documents(query, documents)
        generator_distribution = self.calculate_generator_scores(query, golden_answer, retrieved_documents)
        encoder_log_distribution = torch.softmax(retriever_scores, dim=-1).log()
        loss = kl_div(encoder_log_distribution, generator_distribution, reduction='batchmean')
        return loss

