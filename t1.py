import torch
from torchsummary import summary
from transformers import BertTokenizer, BertModel, BertForMaskedLM
model = BertModel.from_pretrained('bert-base-uncased')

res = summary(model, input_size=(3, 200, 300))
