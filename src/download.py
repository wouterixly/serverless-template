# In this file, we define download_model
# It runs during container build time to get model weights locally

# In this example: A Huggingface BERT model

from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model1 = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./resources/spkrec-ecapa-voxceleb")

    model2 = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir="./resources/wav2vec2-IEMOCAP"
    )

if __name__ == "__main__":
    download_model()

