# Script for training Bart from scratch

This is a minor hack on the Huggingface library to have a script for training Bart from scratch, in particular with the Wikipedia and Books corpus. All the scripts and codes are adapted from the library. Also, it's a simplified version in terms of the noising functions used, where only token masking and sentence permutation are implemented.

## Set up the environment
```
mkdir -p <your_work_dir>
cd <your_work_dir>
conda create -n huggingface_dev python=3.6
conda activate huggingface_dev
pip install torch datasets
git clone --depth 1 --branch v4.0.0 https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
## Replace the code
```
cd <your_work_dir>
git clone https://github.com/jc-hou/train_bart_from_scratch
cd train_bart_from_scratch
cp __init__.py ../transformers/src/transformers/__init__.py
cp data_collator.py ../transformers/src/transformers/data/data_collator.py
cp modeling_bart.py ../transformers/src/transformers/models/bart/modeling_bart.py
```

## Generate the data
```
python ./data/corpus2pkl_bart.py
```
(opt.) To check what the data look like
```
python ./data/decode_pkl.py
```

## Run the script for training Bart from scratch 
```
./run_seq2seqlm.sh
```

## Other
Concerns maybe good to note when trainging bart with the HF library:
- [BartConfig wrong decoder_start_token_id](https://github.com/huggingface/transformers/issues/5212)
- [Bart input format](https://discuss.huggingface.co/t/bart-input-format/1078)

