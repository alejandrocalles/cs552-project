import os
import pandas as pd
from datasets import Dataset

DATAPATH = "./data/m1_preference_data.json"

def create_evaluation_dataset(upload_to: str | None = None) -> None:

    data = pd.read_json(DATAPATH)

    df = data[(data['question_type'] == 'mcq') & (data['question_answer'].map(len) <= 1)].drop(columns=['question_type', 'preferences'])

    df = df[df['question_options'].map(len) == 4]

    df['question_answer'] = df['question_answer'].map({'1': 'A', '2': 'B', '3': 'C', '4': 'D'})

    df.rename(columns={'question_body': 'question', 'question_options' : 'choices', 'question_answer': 'answer'}, inplace=True)

    df['id'] = df['course_id'] * 1000 * 1000 + df['question_id']

    dataset = Dataset.from_pandas(df, preserve_index=False)
    if 'HF_TOKEN' in os.environ:
        token = os.environ['HF_TOKEN']
        dataset.push_to_hub(repo_id='gabrieljimenez/epfl-computer-science-mcqa', split='test', token=token)
    else:
        print("Huggingface token not found in os.environ")

if __name__=="__main__":
    create_evaluation_dataset()