# FlowQA

This code is an example demonstrating how to apply FlowQA model on CoQA Challenge.
We have trained the model with the below config:

python train_CoQA.py --name model_name --epoches 10 --batch_size 2

Achieved f1 score of 77 using dev data

## Steps to run the code
#### Step 1:
Install the required libraries, download the Glove pre-trained embeddings, CoQA train and dev data.
```shell
./download.sh
```
#### Step 2:
Preprocess the data files.
```shell
python preprocess_CoQA.py
```
#### Step 3:
Run the training code.
```shell
python train_CoQA.py --name model_name --epoches 10
```
In the name argument we can give the name of the model to be saved and other arguments can be explored as well.

## Steps to run the code for FlowQA model(lightweight) using small dataset
#### Step 1:
Installs all required files for the experiment.
```shell
./download.sh
```
#### Step 2:
Train_subset.json is a small part of the CoQA train data and is prepared considering the lightweight configuration.
```shell
python preprocess_CoQA.py --train_file train_subset.json --dev_file dev_subset.json
```
#### Step 3:
```shell
python train_CoQA.py --name model_name --epoches 4 --valid_file dev_subset.json
```
####
For the above two methods after training is done, the model and log file are saved in the same directory with the name passed in the "--name" argument.

This can be also run using the collab notebook "FlowQA.ipynb" and the files given. 


References:
1. [FlowQA: Grasping Flow in History for Conversational Machine Comprehension](https://arxiv.org/abs/1810.06683)
2. https://github.com/hsinyuan-huang/FlowQA
