import os
import datasets
import pandas as pd

def create_ms_marco_subset(upload_to: str | None = None) -> None:
    dataset: datasets.DatasetDict = datasets.load_dataset('microsoft/ms_marco', 'v1.1')

    df: pd.DataFrame = dataset['train'].to_pandas()
    df['documents'] = df['passages'].map(lambda passage: passage['passage_text'])
    df['document_mask'] = df['passages'].map(lambda passage: passage['is_selected'])
    df.drop(columns=['passages', 'wellFormedAnswers'], inplace=True)

    N_SAMPLES = 20 * 1000
    RANDOM_SEED = 42
    valid_mask = (df['documents'].map(len) >= 8)

    print(f"Sampling {N_SAMPLES} rows from ms-marco v.1.1")
    clean_df = df[valid_mask].explode('answers').dropna().sample(n=N_SAMPLES, random_state=RANDOM_SEED, ignore_index=True)

    clean_df.head()

    clean_dataset = datasets.Dataset.from_pandas(clean_df, preserve_index=False).rename_column('answers', 'answer')
    final_dataset = clean_dataset.train_test_split(test_size=0.2, shuffle=False)

    if upload_to is None:
        return

    if 'HF_TOKEN' not in os.environ:
        print("Unable to find huggingface token, please ensure the environment variable 'HF_TOKEN' exists and try again.")
        return
        
    final_dataset.push_to_hub(repo_id=upload_to, token=os.getenv('HF_TOKEN'))

if __name__=="__main__":
    create_ms_marco_subset()