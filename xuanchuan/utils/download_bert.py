from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
model.save_pretrained('../bert_model')
tokenizer.save_pretrained('../bert_model')
os.remove("../bert_model/special_tokens_map.json")
os.remove("../bert_model/tokenizer_config.json")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# model.save_pretrained('bert_model_uncased')
# tokenizer.save_pretrained('bert_model_uncased')
