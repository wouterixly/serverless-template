import requests
import torch

sample_rate = 16000
silence = torch.zeros(1, 200*sample_rate)

model_inputs = {'prompt': {'signal': silence.tolist(), 'sr': sample_rate, 'offsets':((1,2),(3,4), (5,6), (6,7), (7,8), (8,9), (9,11),(11,12),(12,14),(14,15),(15,16),(17,19),(19,20),(20,23),(23,24),(24,25))} }

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())
