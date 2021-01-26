from transformers import BertTokenizer, AlbertForMaskedLM
import os
# pretrained = 'voidful/albert_chinese_xlarge'
pretrained = 'voidful/albert_chinese_large'
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = AlbertForMaskedLM.from_pretrained(pretrained)

model.save_pretrained('albert_model')
tokenizer.save_pretrained('albert_model')
# os.remove("albert_model/special_tokens_map.json")
# os.remove("albert_model/tokenizer_config.json")
# os.system("mv albert_model ../")
