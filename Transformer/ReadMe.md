## Introduction
This code is example demonstrating how to apply [Bert](https://arxiv.org/abs/1810.04805) on [CoQA Challenge](https://stanfordnlp.github.io/coqa/).

We have trained the model with the below config:

--type bert \
--bert_model 'bert-large-uncased-whole-word-masking-finetuned-squad' \
--do_train \
--do_predict \
--history_len 2 \
--num_train_epochs 2.0 \
--max_seq_length 512 \
--doc_stride 128 \
--max_query_length 64 \
--output_dir tmp2 \
--train_file coqa-train-v1.0.json \
--predict_file coqa-dev-v1.0.json \
--train_batch_size 12 \
--learning_rate 3e-5 \
--warmup_proportion 0.1 \
--max_grad_norm -1 \
--weight_decay 0.01 \
--fp16 \
--do_lower_case \

On **1x Tesla V100** for **8 Hours** and achieved **82.1 F1-Score** on dev-set.


## Requirement
check requirement.txt or
> pip install -r requirement.txt

## How to run
make sure that:
1. Put *train-coqa-v1.0.json* and *dev-coqa-v1.0.json* on the same dict with *run_coqa.py*
2. The binary file, config file, and vocab file of bert_model in your bert_model dict name as *pytorch_model.bin*, *config.json*, *vocab.txt*
3. Enough memory on GPU [according to this](https://github.com/google-research/bert#out-of-memory-issues), you can tune *--train_batch_size*, *--gradient_accumulation_steps*, *--max_seq_length* and *--fp16* for memeory saving. 

and run
> python run_coqa.py --bert_model your_bertmodel_dir --output_dir your_outputdir \[optional\]

or edit and run *run.sh*

for calculating F1-score, use *evaluate-v1.0.py*
> python evaluate-v1.0.py --data-file <path_to_coqa-dev-v1.0.json> --pred-file <path_to_predictions.json>

# Acknowledgement
We acknowledge the support of Google Developers Expert program for providing GCP credits to carry out the experiments.
