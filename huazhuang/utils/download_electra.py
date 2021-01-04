from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
model.save_pretrained('electra_model')
tokenizer.save_pretrained('electra_model')
os.remove("electra_model/special_tokens_map.json")
os.remove("electra_model/tokenizer_config.json")
os.system("mv electra_model ../")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# model.save_pretrained('bert_model_uncased')
# tokenizer.save_pretrained('bert_model_uncased')
