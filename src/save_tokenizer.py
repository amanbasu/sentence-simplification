'''
Source:
https://huggingface.co/blog/how-to-train
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb#scrollTo=VNZZs-r6iKAV
'''

import datasets
import os

import torch

from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

VOCAB_SIZE = 50000
oscar_path = '../dataset/oscar.en.txt'
tokenizer_path = '../tokenizer'
model_path = '../RoBERTa'

oscar = datasets.load_dataset(
    'oscar',
    'unshuffled_deduplicated_en',
    split='train',
    streaming=True
)

# print('[INFO] reading oscar_en corpus')
# if not os.path.exists(oscar_path) or os.path.getsize(oscar_path) < 1_000_000:
#     with open(oscar_path, 'w') as f:
#         for num, batch in enumerate(oscar):
#             f.write(batch['text'] + '\n')
#             if num > 1_000_000:
#                 break
#     print('[INFO] saved corpora, file size', os.path.getsize(oscar_path))

print('[INFO] training tokenizer')
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[oscar_path], 
    vocab_size=VOCAB_SIZE, 
    min_frequency=2, 
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)
tokenizer.save_model(tokenizer_path)
print('[INFO] saved tokenizer')
