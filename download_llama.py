from huggingface_hub import snapshot_download

HF_AUTH_TOKEN = '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'

snapshot_download(repo_id="meta-llama/Llama-2-13b", local_dir="13B/", use_auth_token=HF_AUTH_TOKEN, local_dir_use_symlinks=False)
