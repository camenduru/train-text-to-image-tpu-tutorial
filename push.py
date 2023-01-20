import gradio as gr
import os
from huggingface_hub import create_repo, upload_folder

block = gr.Blocks()

def build(hf_token):
    repo_id = "camenduru/test-train"
    path_in_repo = ""
    create_repo(repo_id, private=True, token=hf_token)
    upload_folder(folder_path="/home/camenduru/tpu/train", path_in_repo=path_in_repo, repo_id=repo_id, commit_message=f"train", token=hf_token)
    return "done"

def init():
    with block:
        hf_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
        out = gr.Textbox(show_label=False)
        btn = gr.Button("Push to 🤗")
        btn.click(build, inputs=hf_token, outputs=out)
        block.launch(share=True)

if __name__ == "__main__":
    init()
