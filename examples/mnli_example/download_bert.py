from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')