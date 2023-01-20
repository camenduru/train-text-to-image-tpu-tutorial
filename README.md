This repo contains all codes and commands used in `train text to image with tpu tutorial`

## Prepare TPU VM
use with 🐧 linux or linux inside windows (wsl)

https://console.cloud.google.com

```sh
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
gcloud init
```

```sh
gcloud alpha compute tpus tpu-vm ssh node-1 --zone us-central1-f
```

```py
pip install -U zipp "jax[tpu]==0.3.23" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html packaging flax numpy diffusers transformers piexif fold_to_ascii discord ftfy dill urllib3 datasets importlib-metadata accelerate OmegaConf wandb optax torch torchvision modelcards pytorch_lightning protobuf==3.20.* tensorboard markupsafe==2.0.1 gradio
```

```sh
mkdir tpu
cd tpu
mkdir train
cd train
```

```sh
wget https://raw.githubusercontent.com/camenduru/train-text-to-image-tpu-tutorial/main/train_text_to_image_flax.py
```

```sh
tmux
```

## Prepare Dataset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/train-text-to-image-tpu-tutorial/blob/main/prepare_dataset_colab.ipynb)


## Train

```py
python3 train_text_to_image_flax.py \
  --pretrained_model_name_or_path="flax/sd15-non-ema" \
  --dataset_name="camenduru/test" \
  --resolution=512 \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --num_train_epochs=650 \
  --learning_rate=1e-6 \
  --max_grad_norm=1 \
  --output_dir="test" \
  --report_to="wandb"
```

## Push Trained model to 🤗

```sh
wget https://raw.githubusercontent.com/camenduru/train-text-to-image-tpu-tutorial/main/push.py
python3 push.py
```

## Convert Flax model to PyTorch
https://huggingface.co/spaces/camenduru/converter

## Test Flax model or PyTorch model
https://github.com/camenduru/stable-diffusion-diffusers-colab


## Main Repo ♥
https://github.com/huggingface/diffusers

## Scripts From ♥
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py
