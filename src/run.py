# In this file, we define run_model
# It runs every time the server is called

import torch

def run_model(classifier, prompt):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return None
    # do preprocessing
    # prompt is a json of audio array, sample rate, and offsets

    audio = torch.tensor(prompt['signal'])
    sr = prompt['sr']
    offsets = torch.tensor(prompt['offsets'])

    durations = (offsets[:,1] * sr).to(torch.int) - (offsets[:,0] * sr).to(torch.int)

    step = 16
    out = torch.tensor(()).to(device)

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
        out = torch.cat((out, classifier.encode_batch(batch_cuda, signals_batch_lens)), 0 )

    # do postprocessing
    out = torch.squeeze(out).detach().cpu().tolist()

    return out
