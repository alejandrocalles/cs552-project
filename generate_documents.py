import os
import yaml
from tqdm import tqdm
from datasets import load_dataset, Dataset



DATAPATH = "./data/wikipedia-shortlist/computer-science.yml"

def generate_documents(upload_to: str | None = None) -> None:
    with open(DATAPATH, 'r') as file:
        cs_titles_by_subject = yaml.safe_load(file)
    cs_titles = []
    for _, titles in cs_titles_by_subject.items():
        cs_titles += titles
    data = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

    filtered_data = data['train'].filter(lambda row: row['title'] in cs_titles)
    subset = []
    for row in tqdm(filtered_data):
        subset += [row]

    dataset = Dataset.from_list(subset).rename_column("url", "source")

    if upload_to is None:
        return
    
    if 'HF_TOKEN' not in os.environ:
        return
    
    dataset.push_to_hub(repo_id=upload_to, split='train', token=os.environ['HF_TOKEN'])

if __name__=="__main__":
    generate_documents()
