# In this file, we define load_model
# It runs once at server startup to load the model to a GPU

import torch
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class

def load_model():

    # obtain device for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return None

    # load the model from cache or local file to the GPU
    model1 = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./resources/spkrec-ecapa-voxceleb", run_opts={"device": device})

    model2 = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir="./resources/wav2vec2-IEMOCAP",
        run_opts={"device": device}
    )

    # return the callable model
    return model1, model2

