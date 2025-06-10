import os
from huggingface_hub import HfApi

HF_USERNAME = "gabrieljimenez"

def get_token():
  if 'HF_TOKEN' in os.environ:
    return os.getenv('HF_TOKEN')
  else:
    from google.colab import userdata
    return userdata.get('HF_TOKEN')

def upload(folder_name, repo_name, is_model_repo):
  api = HfApi(token = get_token())
  repo_id = HF_USERNAME + "/" + repo_name
  absolute_path = "/content/" + folder_name
  print(f"Uploading the following files from {absolute_path} to repository {repo_id}")
  print("\n".join(os.listdir(absolute_path)))
  api.upload_folder(
    folder_path=absolute_path,
    repo_id=repo_id,
    repo_type="model" if is_model_repo else "dataset",
  )

def upload_rag_model_from(folder_name):
  upload(
    folder_name=folder_name,
    repo_name="MNLP_M3_rag_model",
    is_model_repo=True,
  )

def upload_encoder_model_from(folder_name):
  upload(
    folder_name=folder_name,
    repo_name="MNLP_M3_document_encoder",
    is_model_repo=True,
  )