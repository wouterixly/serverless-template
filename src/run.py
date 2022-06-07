# In this file, we define run_model
# It runs every time the server is called

import pickle
import torch
import requests
from sklearn.cluster import KMeans

def run_model(model1, model2, url):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return None
    # do preprocessing

    response = pickle.loads(requests.get(url).content)

    audio = response['training_data']['interview_data']
    sr = response['training_data']['sample_rate']
    offsets = list(response['asr_df']['offsets'].values())
    n_clusters = len(response['training_data']['speakers'])

    offsets = torch.tensor(offsets)

    model1_out = torch.tensor(()).to(device)
    model2_out = []

    for offset in offsets:

        signal_piece = audio[:, int( offset[0] * sr) : int( offset[1] * sr ) ].to(device)

        model1_out = torch.cat((model1_out, model1.encode_batch(signal_piece)), 0 )
        model2_out.extend(model2.classify_batch(signal_piece)[3])
        
    # do postprocessing
    

    xvectors = torch.squeeze(model1_out).detach().cpu()
    kmeans = KMeans(n_clusters=n_clusters).fit(xvectors)

    return {'clusters':kmeans.labels_.tolist(), 'emos':model2_out}
