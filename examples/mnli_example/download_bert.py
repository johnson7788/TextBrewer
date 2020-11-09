from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model.save_pretrained('bert-base-uncased')
tokenizer.save_pretrained('bert-base-uncased')