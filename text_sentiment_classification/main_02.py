from transformers import BertModel, BertTokenizer

# https://huggingface.co/bert-base-uncased#how-to-use
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    cache_dir="tokenizers",
)
model = BertModel.from_pretrained(
    "bert-base-uncased",
    cache_dir="models",
)
text = r"Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)


# https://www.analyticsvidhya.com/blog/2021/09/an-explanatory-guide-to-bert-tokenizer/
encoding = tokenizer.encode(text)
tokenized = tokenizer.convert_ids_to_tokens(encoding)
"breakpoint"

# TODO READ:
# https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/fastai/bert_with_fastai.ipynb
# https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2
# https://www.kaggle.com/code/maroberti/fastai-with-transformers-bert-roberta
# https://github.com/utterworks/fast-bert
