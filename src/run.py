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
    durations = (offsets[:,1] * sr).to(torch.int) - (offsets[:,0] * sr).to(torch.int)

    step = 16
    model1_out = torch.tensor(()).to(device)
    model2_out = []

    for i in range(0,len(offsets), step):
    
        #change step size at last instance
        if i >= len(offsets)-step:
            step = len(offsets[i :])
    
        local_max = int(max(durations[i : i + step]))
    
        signals_batch = torch.tensor(())
        signals_batch_lens = torch.tensor(())
    
        #pad the signals to the max of the batch and encode
        for k in range(i , i + step):
            signal_piece = audio[:, int( offsets[k, 0] * sr) : int( offsets[k, 1] * sr ) ]

            signals_batch = torch.cat((signals_batch,
                                       torch.cat((signal_piece, torch.zeros(1, local_max - durations[k])), 1)
                                      ), 0 )

        signals_batch_lens = durations[i:i+step]/local_max

        batch_cuda = signals_batch.to(device)
        model1_out = torch.cat((model1_out, model1.encode_batch(batch_cuda, signals_batch_lens)), 0 )
        model2_out.extend(model2.classify_batch(batch_cuda, signals_batch_lens)[3])     

    # do postprocessing
    

    xvectors = torch.squeeze(model1_out).detach().cpu()
    kmeans = KMeans(n_clusters=n_clusters).fit(xvectors)

    return {'clusters':kmeans.labels_.tolist(), 'emos':model2_out}
