from tqdm import tqdm
import sari
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import GPT2TokenizerFast, GPT2Config, EncoderDecoderConfig, EncoderDecoderModel
from torch.utils.data import DataLoader
from datagen import DataGenerator
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

tokenizer = None
MAX_LENGTH = 100
BATCH_SIZE = 20
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = '../checkpoint/model_gpt2.pt'
PRED_PATH = '../gpt2_preds_test.txt'
softmax = torch.nn.LogSoftmax(dim=-1)

def bleu_score(pred, ref):
    weights = (0.25, 0.25, 0.25, 0.25)
    chencherry = SmoothingFunction().method1
    
    score = 0
    for a, b in zip(ref, pred):
        for idx, sent in enumerate(a):
            a[idx] = sent.split(' ')
        b = b.split(' ')
        score += corpus_bleu([a], [b], weights=weights, smoothing_function=chencherry)

    return score/BATCH_SIZE

def sari_score(src, pred, ref):
    score = 0
    for a, b, c in zip(src, pred, ref):
        score += sari.SARIsent(a, b, c)
    return score/BATCH_SIZE

def get_dataloader():
    testDataset = DataGenerator(mode='test')
    testLoader = DataLoader(
        testDataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )

    return iter(testLoader)

def encode_batch(src, tgt, max_len=100):
    src_tok = tokenizer(
        src, 
        max_length=max_len, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    ).to(DEVICE)

    tgt_tok = tokenizer(
        tgt, 
        max_length=max_len, 
        padding='max_length',
        truncation=True, 
        return_tensors='pt'
    ).to(DEVICE)

    labels = tgt_tok.input_ids.clone()
    labels[tgt_tok.attention_mask == 0] = -100

    return src_tok.input_ids, src_tok.attention_mask, tgt_tok.input_ids, tgt_tok.attention_mask, labels

def eval(model):

    testLoader = get_dataloader()

    predictions = []

    # get model performace on val set
    with torch.no_grad():
        losses, bleus, saris = [], [], []
        try:
            for source, target, ref in tqdm(testLoader):

                ref = np.array(ref).T.tolist()                                  # transpose ref, order gets changed in datagen

                src_inp_ids, src_att_mask, tgt_inp_ids, tgt_att_mask, labels = encode_batch(
                    source, target
                )

                loss, logits = model(
                    input_ids = src_inp_ids, 
                    decoder_input_ids = tgt_inp_ids,
                    attention_mask = src_att_mask,
                    decoder_attention_mask = tgt_att_mask,
                    labels = labels
                )[:2]

                outputs = torch.argmax(softmax(logits), dim=-1) 
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # for idx in range(len(outputs)):
                #     target_split = target[idx].strip().split('.')
                #     nsent = len(target_split)
                #     if target_split[-1] == '':
                #         nsent -= 1

                #     outputs[idx] = '.'.join(outputs[idx].split('.')[:nsent]) + '.'
                
                predictions.extend(outputs)

                sari = sari_score(source, outputs, ref)         
                bleu = bleu_score(outputs, ref)

                losses += [loss.item()]
                saris += [sari]
                bleus += [bleu]

        except StopIteration:
            pass

    print(f'loss: {np.mean(losses):.4f} - sari: {np.mean(saris):.4f} - bleu: {np.mean(bleus):.4f}')
    # print(f"{idx//BATCH_SIZE+1}/{SIZE//BATCH_SIZE} [{'=' * progress}>{'-' * (nlines - progress)}] loss: {np.mean(losses):.3f}", end='\r')

    return predictions        

if __name__ == '__main__':

    print('using device:', DEVICE)
    print('loading from:', SAVE_PATH)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    gpt2config = GPT2Config()
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        gpt2config, gpt2config
    )
    model = EncoderDecoderModel(config=config)

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = MAX_LENGTH
    model.config.no_repeat_ngram_size = 3
    model = model.to(DEVICE)

    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
        
    predictions = eval(model)

    # with open(PRED_PATH, 'w') as f:
    #     for pred in predictions:
    #         f.write(pred + '\n')