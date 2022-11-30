from tqdm import tqdm
import torch
import numpy as np
import os
from utils import encode_batch, get_testloader, sari_score, bleu_score, select_model

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

mod = 'bert'
encoderTokenizer, decoderTokenizer = None, None
MAX_LENGTH = 100
BATCH_SIZE = 20
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = f'../checkpoint/model_{mod}.pt'
PRED_PATH = f'../{mod}_preds_test.txt'
softmax = torch.nn.LogSoftmax(dim=-1)

def eval(model):

    testLoader = get_testloader(BATCH_SIZE)

    predictions = []

    # get model performace on val set
    with torch.no_grad():
        losses, bleus, saris = [], [], []
        try:
            for source, target, ref in tqdm(testLoader):

                ref = np.array(ref).T.tolist()                                  # transpose ref, order gets changed in datagen

                src_inp, src_att, tgt_inp, tgt_att, labels = encode_batch(
                    encoderTokenizer, decoderTokenizer, source, target
                )

                loss, logits = model(
                    input_ids = src_inp.to(DEVICE), 
                    decoder_input_ids = tgt_inp.to(DEVICE),
                    attention_mask = src_att.to(DEVICE),
                    decoder_attention_mask = tgt_att.to(DEVICE),
                    labels = labels.to(DEVICE)
                )[:2]

                outputs = torch.argmax(softmax(logits), dim=-1) 
                outputs = decoderTokenizer.batch_decode(outputs, skip_special_tokens=True)

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

    encoderTokenizer, decoderTokenizer, model = select_model(mod=mod)

    model.config.max_length = MAX_LENGTH
    model.config.no_repeat_ngram_size = 3
    model = model.to(DEVICE)

    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
        
    predictions = eval(model)

    with open(PRED_PATH, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')