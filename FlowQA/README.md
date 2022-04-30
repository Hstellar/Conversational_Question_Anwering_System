# FlowQA

This code is an example demonstrating how to apply FlowQA model on CoQA Challenge.
We have trained the model with the below config:

python train_CoQA.py --name model_name --epoches 10 --batch_size 2

Achieved f1 score of 77 using dev data

## Steps to run the code
#### Step 1:
```shell
./download.sh
```
--> Run the above command to install the required libraries, download the Glove pre-trained embeddings, CoQA train and dev data.

#### Step 2:
preprocess the data files using command:
```shell
python preprocess_CoQA.py
```
#### Step 3:
run the training code using:
```shell
python train_CoQA.py --name model_name --epoches 10
```
In the name argument we can give the name of the model to be saved and other arguments can be explored as well.

## Steps to run the code for FlowQA model(lightweight) using small dataset
#### Step 1:
```shell
./download.sh
```
Installs all required files for the experiment.
#### Step 2:
```shell
python preprocess_CoQA.py --train_file train_subset.json --dev_file dev_subset.json
```
-->train_subset.json is a small part of the CoQA train data which has been taken based on instructor's requirements.Similar idea is replicated for valid data as well.

#### Step 3:
```shell
python train_CoQA.py --name model_name --epoches 4 --valid_file dev_subset.json
```
####

For the above two methods after training the model and log file are saved in the same directory with the name passed in the "--name" argument.

This can be also run using the collab notebook "FlowQA.ipynb" and the files given 


References:
1. [FlowQA: Grasping Flow in History for Conversational Machine Comprehension](https://arxiv.org/abs/1810.06683)
2. https://github.com/hsinyuan-huang/FlowQA
