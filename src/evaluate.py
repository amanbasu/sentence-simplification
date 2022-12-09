from tqdm import tqdm
import torch
import numpy as np
import os
from utils import *
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

encoderTokenizer, decoderTokenizer = None, None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval(model, args):
    global encoderTokenizer, decoderTokenizer, DEVICE

    testLoader = get_testloader(args.batch_size)
    predictions = []
    metric = {'sari': [], 'bleu': [], 'fkgl': []}

    # get model performace on val set
    with torch.no_grad():
        try:
            for source, target, ref in tqdm(testLoader):
                ref = np.array(ref).T.tolist()                                  # transpose ref, order gets changed in datagen

                src_inp, _, _, _, _ = encode_batch(
                    encoderTokenizer, decoderTokenizer, source, target
                )

                logits = model.generate(
                    input_ids=src_inp.to(DEVICE), 
                    max_length=args.max_length
                )
                outputs = decoderTokenizer.batch_decode(
                    logits, skip_special_tokens=True
                )

                predictions.extend(outputs)

                sari = sari_score(source, outputs, ref)         
                bleu = bleu_score(outputs, ref)
                fkgl = fkgl_score(outputs)

                metric['sari'] += sari
                metric['bleu'] += bleu
                metric['fkgl'] += fkgl
        except StopIteration:
            pass

    log = []
    for key in metric.keys():
        log.append(f'{key}: {np.mean(metric[key]):.4f}')
    print(' - '.join(log))

    return predictions

def main(args):
    global encoderTokenizer, decoderTokenizer, DEVICE

    print('using device:', DEVICE)
    print('loading from:', args.model_path)

    encoderTokenizer, decoderTokenizer, model = select_model(mod=args.model)

    model.config.max_length = args.max_length
    model.config.no_repeat_ngram_size = 3
    model = model.to(DEVICE)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
            
    predictions = eval(model, args)

    if args.save_predictions:
        with open(args.pred_path, 'w') as f:
            for pred in predictions:
                f.write(pred + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for evaluation.')
    parser.add_argument(
        '--model', default='gpt2', type=str, 
        choices=['gpt2', 'bert', 'bert_gpt2', 'gpt2_bert'],
        help='model type'
    )
    parser.add_argument(
        '--max_length', default=80, type=int,
        help='maximum length for encoder'
    )
    parser.add_argument(
        '--batch_size', default=20, type=int,
        help='batch size for evaluation'
    )
    parser.add_argument(
        '--model_path', default='../checkpoint/model.pt', type=str,
        help='model save path'
    )
    parser.add_argument(
        '--save_predictions', default=False, type=bool,
        choices=[True, False],
        help='saves predictions in a txt file'
    )
    parser.add_argument(
        '--pred_path', default='predictions.txt', type=str,
        help='path to save the predictions'
    )
    args = parser.parse_args()
    main(args)