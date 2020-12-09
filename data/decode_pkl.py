import pickle

pkl_save_name = "wikipedia_bookcorpus_bart_blocksize_1024_concate.pkl"
with open(pkl_save_name, 'rb') as f:
    pkl_data = pickle.load(f)
train_dataset = pkl_data['tokenized_datasets']

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
encoded_sequence = train_dataset[0]['input_ids'] # should from wikipedia 
decoded_sequence = tokenizer.decode(encoded_sequence)
print("encoded sequence: ", encoded_sequence)
print("decoded sequence: ", decoded_sequence)
print("\n\n\n\n\n\n\n")
encoded_sequence = train_dataset[-1]['input_ids'] # should from bookcorpus 
decoded_sequence = tokenizer.decode(encoded_sequence)
print("encoded sequence: ", encoded_sequence)
print("decoded sequence: ", decoded_sequence)
