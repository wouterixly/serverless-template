# In this file, we define load_model
# It runs once at server startup to load the model to a GPU

import torch
from speechbrain.pretrained import EncoderClassifier

def load_model():

    # obtain device for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return None

    # load the model from cache or local file to the GPU
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./resources/spkrec-ecapa-voxceleb", run_opts={"device": device})

    # return the callable model
    return classifier

