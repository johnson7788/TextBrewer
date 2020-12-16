from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base")
model.save_pretrained('mac_bert_model')
tokenizer.save_pretrained('mac_bert_model')
os.remove("mac_bert_model/special_tokens_map.json")
os.remove("mac_bert_model/tokenizer_config.json")
os.system("mv mac_bert_model ../")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# model.save_pretrained('bert_model_uncased')
# tokenizer.save_pretrained('bert_model_uncased')
