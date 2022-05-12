# In this file, we define download_model
# It runs during container build time to get model weights locally

# In this example: A Huggingface BERT model

from speechbrain.pretrained import EncoderClassifier


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./resources/spkrec-ecapa-voxceleb")

if __name__ == "__main__":
    download_model()

