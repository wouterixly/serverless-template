import requests
import torch

sample_rate = 16000
silence = torch.zeros(1, 200*sample_rate)

model_inputs = {'s3_url':'https://ixly-datascience.s3.amazonaws.com/ixly_nlp_app/transcribed/1e277ed7-a228-4c63-bc33-b560afc4be89_transcribed.pkl?AWSAccessKeyId=AKIASQYHVZZCI6FMCOH6&Signature=4F88rRebgTBodDKb9fWlKhyXxO8%3D&Expires=1654093265'}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())
