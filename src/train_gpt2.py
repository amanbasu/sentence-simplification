from tqdm import tqdm
import sari
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from transformers import GPT2TokenizerFast, GPT2Config, EncoderDecoderConfig, EncoderDecoderModel
from torch.utils.data import DataLoader
from datagen import DataGenerator
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

tokenizer = None
nlines = 20
MAX_LENGTH = 100
BATCH_SIZE = 50
INIT_EPOCH = 0
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = '../checkpoint/model_gpt2.pt'
softmax = torch.nn.LogSoftmax(dim=-1)

def bleu_score(pred, ref):
    weights=(0.5, 0.5,)
    return corpus_bleu(ref, pred, weights)

def sari_score(src, pred, ref):
    score = 0
    for a, b, c in zip(src, pred, ref):
        score += sari.SARIsent(a, b, c)
    return score/BATCH_SIZE

def get_dataloader():
    trainDataset = DataGenerator(mode='train')
    valDataset = DataGenerator(mode='valid')

    trainLoader = DataLoader(
        trainDataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    valLoader = DataLoader(
        valDataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )
    return iter(trainLoader), iter(valLoader)

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

def train(model, optimizer):

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        trainLoader, valLoader = get_dataloader()

        # when dataloader runs out of batches, it throws an exception
        try:
            for source, target in tqdm(trainLoader):
                src_inp_ids, src_att_mask, tgt_inp_ids, tgt_att_mask, labels = encode_batch(
                    source, target
                )

                optimizer.zero_grad(set_to_none=True)                           # clear gradients w.r.t. parameters

                loss = model(
                    input_ids = src_inp_ids, 
                    decoder_input_ids = tgt_inp_ids,
                    attention_mask = src_att_mask,
                    decoder_attention_mask = tgt_att_mask,
                    labels = labels
                )[0]

                loss.backward()                                                 # getting gradients
                optimizer.step()                                                # updating parameters
        except StopIteration:
            pass

        # get model performace on val set
        with torch.no_grad():
            losses, bleus, saris = [], [], []
            try:
                for source, target, ref in tqdm(valLoader):

                    ref = np.array(ref).T.tolist()                              # transpose ref, order gets changed in datagen

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

                    sari = sari_score(source, outputs, ref)         
                    bleu = bleu_score(outputs, ref)

                    losses += [loss.item()]
                    saris += [sari]
                    bleus += [bleu]

            except StopIteration:
                pass

        print(f'loss: {np.mean(losses):.4f} - sari: {np.mean(saris):.4f} - bleu: {np.mean(bleus):.4f}')
        # print(f"{idx//BATCH_SIZE+1}/{SIZE//BATCH_SIZE} [{'=' * progress}>{'-' * (nlines - progress)}] loss: {np.mean(losses):.3f}", end='\r')

        # save model checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, f'../checkpoint/model_{epoch+1}.pt')
        print('checkpoint saved.')

    return model        

if __name__ == '__main__':

    print('using device:', DEVICE)
    print('save path:', SAVE_PATH)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # start from last checkpoint
    if INIT_EPOCH > 0:
        print('loading from', SAVE_PATH)
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        INIT_EPOCH = checkpoint['epoch']
        
    model = train(model, optimizer)

    torch.save(
        {
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVE_PATH)
    print('model saved.')