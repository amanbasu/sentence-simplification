# Sentence Simplification

<img src="https://img.shields.io/badge/torch-1.10.2+cu102-green?logo=pytorch"/> <img src="https://img.shields.io/badge/python-3.9.8-blue?logo=python"/>

> This work is part of the course L-645 (CSCI B-659) Advanced Natural Language Processing at Indiana University, Bloomington.

Paper link: https://arxiv.org/abs/2212.04595

## âšī¸ Overview

Sentence simplification aims at making the structure of text easier to read and understand while maintaining its original meaning. This can be helpful for people with disabilities, new language learners, or those with low literacy. Simplification often involves removing difficult words and rephrasing the sentence. This repo contains the code for fine-tuning transformer models for the task of sentence simplification.

## đ Highlights

- Train models for sentence simplification in PyTorch.
- Models include GPT-2, BERT, GPT-2 encoder and BERT decoder, BERT encoder and GPT-2 decoder.
- Evaluate results on metrics like SARI, FKGL, and BLEU.

## đ Dataset

The models were trained on *WikiLarge* dataset which you can download from David Kauchak's [webpage](https://cs.pomona.edu/~dkauchak/simplification/) or [XingxingZhang/dress](https://github.com/XingxingZhang/dress).

> For ease of access, a small, cleaned dataset from [Aakash12980/Sentence-Simplification-using-BERT-GPT2](https://github.com/Aakash12980/Sentence-Simplification-using-BERT-GPT2) is provided. This dataset can also be used for training and evaluation purpose.

## đī¸ Folder structure

```
.
â
âââ dataset
â   âââ src_train.txt
â   âââ src_valid.txt
â   âââ src_test.txt
â   âââ tgt_train.txt
â   âââ tgt_valid.txt
â   âââ tgt_test.txt
â   âââ ref_test.pkl
â   âââ ref_valid.pkl
â
âââ src
â   âââ datagen.py
â   âââ evaluate.py
â   âââ sari.py
â   âââ train.py
â   âââ train_from_scratch.py
â   âââ utils.py
|
âââ requirements.txt
âââ README.md
|
```

## â Requirements

The code uses python `3.9.8` and torch `1.10.2`. The other requirements are:

```bash
nltk==3.6.7
numpy==1.22.2
tokenizers==0.13.2
torch==1.10.2
tqdm==4.62.3
transformers==4.24.0
```

## đ¨âđģ Usage

Download the code

```bash
https://github.com/amanbasu/sentence-simplification.git
```

Install requirements

```bash
pip install -r requirements.txt
```

train.py usage

```bash
$ python train.py -h
usage: train.py [-h] [--model {gpt2,bert,bert_gpt2,gpt2_bert}] [--max_length MAX_LENGTH] [--epochs EPOCHS] [--init_epoch INIT_EPOCH] [--batch_size BATCH_SIZE] [--lr LR] [--save_path SAVE_PATH]

Arguments for training.

optional arguments:
  -h, --help            show this help message and exit
  --model {gpt2,bert,bert_gpt2,gpt2_bert}
                        model type
  --max_length MAX_LENGTH
                        maximum length for encoder
  --epochs EPOCHS       number of training epochs
  --init_epoch INIT_EPOCH
                        epoch to resume the training from
  --batch_size BATCH_SIZE
                        batch size for training
  --lr LR               learning rate for training
  --save_path SAVE_PATH
                        model save path
```

evaluate.py usage

```bash
$ python evaluate.py -h
usage: evaluate.py [-h] [--model {gpt2,bert,bert_gpt2,gpt2_bert}] [--max_length MAX_LENGTH] [--batch_size BATCH_SIZE] [--model_path MODEL_PATH] [--save_predictions {True,False}] [--pred_path PRED_PATH]

Arguments for evaluation.

optional arguments:
  -h, --help            show this help message and exit
  --model {gpt2,bert,bert_gpt2,gpt2_bert}
                        model type
  --max_length MAX_LENGTH
                        maximum length for encoder
  --batch_size BATCH_SIZE
                        batch size for evaluation
  --model_path MODEL_PATH
                        model save path
  --save_predictions {True,False}
                        saves predictions in a txt file
  --pred_path PRED_PATH
                        path to save the predictions
```

## đŦ Examples

To train a model

```bash
python train.py --model bert --epochs 5 --batch_size 20 --save_path '../checkpoint/model_bert.pt'
```

To evaluate a model

```bash
python evaluate.py --model bert --model_path '../checkpoint/model_bert.pt' --save_predictions True --pred_path '../bert_predictions.txt'
```

## đ Results

<p align='center'><img src="results.png" width="50%"></p>
<p align='center'>Figure 1. A comparison of our model's performance against previous studies.</p>

## âī¸ Citation

```
@inproceedings{Agarwal2022ExplainTM,
  title={Explain to me like I am five -- Sentence Simplification Using Transformers},
  author={Aman Agarwal},
  year={2022},
  doi = {10.48550/ARXIV.2212.04595},
}
```

## đ References

1. Sample data: https://github.com/Aakash12980/Sentence-Simplification-using-BERT-GPT2
2. WikiLarge: https://cs.pomona.edu/~dkauchak/simplification/
3. Train models from scratch: https://huggingface.co/blog/how-to-train
4. SARI implementation: https://github.com/cocoxu/simplification
5. Other metrics (EASSE): https://github.com/feralvam/easse
