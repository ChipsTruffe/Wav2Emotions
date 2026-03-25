from datasets import load_dataset

ds = load_dataset("AbstractTTS/IEMOCAP")
print(ds)
print(ds["train"][0].keys())