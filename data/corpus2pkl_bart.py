# Adated from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py
import pickle
from datasets import load_dataset, concatenate_datasets
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


wiki = load_dataset("wikipedia", "20200501.en", split="train") # len(wiki):6078422
bookcorpus = load_dataset("bookcorpus", split="train") # len(bookcorpus): 74004228

print(wiki.column_names, bookcorpus.column_names)
# ['title', 'text'] ['text']
wiki.remove_columns_("title")
bart_dataset = concatenate_datasets([wiki, bookcorpus]) # len(bart_dataset):80082650

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large', use_fast=True)

# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = bart_dataset.column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_datasets = bart_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=16,
    remove_columns=column_names,
    load_from_cache_file=False,
)

block_size = 1024 # tokenizer.model_max_length

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=16,
    load_from_cache_file=False,
)

pkl_save_name = "wikipedia_bookcorpus_bart_blocksize_1024_concate.pkl"
with open(pkl_save_name, 'wb') as handle:
    pickle.dump({'tokenized_datasets': lm_datasets}, handle, protocol=pickle.HIGHEST_PROTOCOL)



