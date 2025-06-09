import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity
from torch.nn import CrossEntropyLoss

class RAGPipeline:

    def __init__(self, model, tokenizer, encoder_model, encoder_tokenizer, k=1, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.encoder_model = encoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.cuda = cuda
        self.model_optimizer = Adam(self.model.parameters(), lr=1e-5)
        self.encoder_optimizer = Adam(self.encoder_model.parameters(), lr=1e-5)

    def retrieve_documents(self, query: str, documents: list[str]):
        if len(documents) < self.k:
            raise Error("Invalid input: number of candidate documents is smaller than retrieval 'k'.")
        tokenized_query = self.encoder_tokenizer(query, return_tensors="pt")
        tokenized_documents = [self.encoder_tokenizer(document, return_tensors="pt") for document in documents]

        embedded_query = self.encoder_model(**tokenized_query)['pooler_output']
        embedded_documents = [self.encoder_model(**document)['pooler_output'] for document in tokenized_documents]

        scores = torch.tensor([cosine_similarity(document, embedded_query) for document in embedded_documents])

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

    def calculate_loss(self, query: str, golden_answer: str, background: list[str]):
        augmented_prompt = self.build_prompt(query, background)

        tokenized_inputs = self.tokenizer(text=augmented_prompt, text_target=golden_answer, return_tensors="pt")
        inputs = dict(tokenized_inputs)
        inputs['input_ids'] = torch.cat([tokenized_inputs['input_ids'], tokenized_inputs['labels']], dim=1)
        inputs['labels'] = inputs['input_ids'].clone()
        inputs['labels'][:, :tokenized_inputs['input_ids'].size()[1]] = -100
        inputs['attention_mask'] = torch.ones(inputs['input_ids'].size())
        inputs['attention_mask'][:, :tokenized_inputs['attention_mask'].size()[1]] = tokenized_inputs['attention_mask']

        if self.cuda:
            # Move all tensors to GPU
            inputs['input_ids'] = inputs['input_ids'].to('cuda')
            inputs['labels'] = inputs['labels'].to('cuda')
            inputs['attention_mask'] = inputs['attention_mask'].to('cuda')

        outputs = self.model(**inputs)

        if self.cuda:
            loss = outputs['loss'].to('cpu')
        else:
            loss = outputs['loss']

        if self.cuda:
            # Remove all tensors from GPU
            del inputs['input_ids']
            del inputs['labels']
            del inputs['attention_mask']

        return loss

    def calculate_losses(self, query: str, golden_answer: str, retrieved_documents: list[str]):
        augmented_prompts = [self.build_prompt(query, [document]) for document in retrieved_documents]

        tokenized_prompts = self.tokenizer(text=augmented_prompts, padding=False)
        tokenized_golden_answer = self.tokenizer(text=golden_answer, padding=False)

        # Save the state of the warning, then disable it
        warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        try:
            inputs = tokenizer.pad(
                encoded_inputs={
                    'input_ids': [
                        tokenized_prompt + tokenized_golden_answer for tokenized_prompt in tokenized_prompts
                    ]
                },
                padding=True,
                return_tensors="pt"
            )
            labels = tokenizer.pad(
                encoded_inputs={
                    'input_ids': [
                        [-100] * len(tokenized_prompt) + tokenized_golden_answer for tokenized_prompt in tokenized_prompts
                    ]
                },
                padding=True,
                return_tensors="pt"
            )
        finally:
            # Restore the state of the warning.
            tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

        labels['input_ids'][labels['attention_mask'] == 0] = -100
        inputs['labels'] = labels['input_ids']

        if self.cuda:
            # Move all tensors to GPU
            inputs['input_ids'] = inputs['input_ids'].to('cuda')
            inputs['labels'] = inputs['labels'].to('cuda')
            inputs['attention_mask'] = inputs['attention_mask'].to('cuda')

        outputs = self.model(**inputs)

        shifted_logits = outputs['logits'][:, :-1, :]
        shifted_labels = labels['input_ids'][:, 1:]

        loss_function = CrossEntropyLoss(reduction="none", ignore_index=-100)
        losses_per_label = loss_function(
            torch.transpose(shifted_logits, -2, -1),
            shifted_labels
        )

        shifted_mask = (shifted_labels != -100)
        losses = torch.sum(losses_per_label * shifted_mask, dim=1) / torch.sum(shifted_mask, dim=1)

        if self.cuda:
            losses = losses.to('cpu')

        return losses

    def calculate_scores(self, query: str, golden_answer: str, retrieved_documents: list[str], temperature: float = 0.01):
        losses = []
        for document in retrieved_documents:
            losses += [self.calculate_loss(query, golden_answer, [document])]

        losses = torch.stack(losses)
        # Normalize to get p(c|x,y)
        scores = torch.softmax(-losses / temperature, dim=-1)
        return scores
