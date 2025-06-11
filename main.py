import sys
import logging
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from pipeline import RAGPipeline
from radit import RADITTrainer

from ms_marco import create_ms_marco_subset
from create_evaluation_dataset import create_evaluation_dataset
from generate_documents import generate_documents


logger = logging.getLogger(__name__)


BASE_MODEL_NAME = "Hyeongdon/MNLP_M3_sft_model"
BASE_ENCODER_NAME = "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT"
TRAIN_DATASET_NAME = "gabrieljimenez/MNLP_M3_rag_dataset"


def main(argv: list[str]):

    logger.info("Generating training set...")

    create_ms_marco_subset(upload_to=None)

    logger.info("Generating evaluation set...")

    try:
        create_evaluation_dataset()
    except OSError:
        logger.warning("Creation of evaluation data failed, probably because the source file was not found.")

    logger.info("Generating document set...")

    generate_documents()

    HAS_CUDA = torch.cuda.is_available()

    logger.info("Loading models...")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    encoder_model = SentenceTransformer(BASE_ENCODER_NAME)

    if HAS_CUDA:
        model.to('cuda')
        encoder_model.to('cuda')
    

    logger.info("Loading dataset...")

    data = datasets.load_dataset(TRAIN_DATASET_NAME)

    pipeline = RAGPipeline(model, tokenizer, encoder_model, k=3, cuda=HAS_CUDA, lr=1e-6)

    trainer = RADITTrainer(pipeline, train_dataset=data['train'], test_dataset=data['test'], epochs=1, batch_size=100)

    logger.info("Starting training...")

    trainer.train(output_filepath='./outputs/training.out')

    if '--save-models' in argv[1:]:
        logger.info("Saving models...")
        model.save_pretrained("./model/")
        tokenizer.save_pretrained("./model/")

        encoder_model.save_pretrained("./encoder_model/")



if __name__ == "__main__":
    main(sys.argv)
