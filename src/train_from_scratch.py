'''
Source:
https://huggingface.co/blog/how-to-train
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb#scrollTo=VNZZs-r6iKAV

The file trains a RoBERTa model from scratch on oscar.en data from HuggingFace.
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

print('[INFO] reading oscar_en corpus')
if not os.path.exists(oscar_path) or os.path.getsize(oscar_path) < 1_000_000:
    with open(oscar_path, 'w') as f:
        for num, batch in enumerate(oscar):
            f.write(batch['text'] + '\n')
            if num > 1_000_000:
                break
    print('[INFO] saved corpora, file size', os.path.getsize(oscar_path))

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

tokenizer = ByteLevelBPETokenizer(
    f'{tokenizer_path}/vocab.json', f'{tokenizer_path}/merges.txt'
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

config = RobertaConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=512)

model = RobertaForMaskedLM(config=config)
print('[INFO] model parameters:', model.num_parameters())

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=oscar_path,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print(f'[INFO] training RoBERTa on gpu: {torch.cuda.is_available()}')
trainer.train()
trainer.save_model(model_path)
print('[INFO] model saved')

fill_mask = pipeline(
    "fill-mask",
    model=model_path,
    tokenizer=tokenizer_path
)

print('[INFO] sanity check')
print(fill_mask('Let children play <mask>.'))
print(fill_mask('Sun rises in the <east>.'))
print(fill_mask('David went to a <mask> store to buy the toilet paper.'))
print('done')