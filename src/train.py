from tqdm import tqdm
import torch
import numpy as np
import os
from utils import *
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
encoderTokenizer, decoderTokenizer = None, None

def train(model, optimizer, args):
    global encoderTokenizer, decoderTokenizer, DEVICE

    scores = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        trainLoader, valLoader = get_dataloader(args.batch_size)
        metric = {'loss': [], 'sari': [], 'bleu': [], 'fkgl': []}
        
        # when dataloader runs out of batches, it throws an exception
        try:
            for source, target in tqdm(trainLoader):
                src_inp, _, _, _, labels = encode_batch(
                    encoderTokenizer, decoderTokenizer, source, target
                )

                optimizer.zero_grad(set_to_none=True)

                loss = model(
                    input_ids = src_inp.to(DEVICE), 
                    labels = labels.to(DEVICE)
                )[0]

                metric['loss'] += [loss.item()]
                loss.backward()
                optimizer.step()
                # scheduler.step()   
        except StopIteration:
            pass

        # get model performace on val set
        with torch.no_grad():
            try:
                for source, target, ref in tqdm(valLoader):
                    ref = np.array(ref).T.tolist()                              # transpose ref, order gets changed in datagen

                    src_inp, _, _, _, labels = encode_batch(
                        encoderTokenizer, decoderTokenizer, source, target
                    )

                    logits = model.generate(
                        input_ids=src_inp.to(DEVICE), 
                        max_length=args.max_length
                    )
                    outputs = decoderTokenizer.batch_decode(
                        logits, skip_special_tokens=True
                    )

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

        scores.append(np.mean(metric['sari']))
        # save checkpoint for only the best model 
        if epoch > 1 and scores[-1] == np.max(scores):
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, f'../checkpoint/model_{args.model}_best.pt')
            print('checkpoint saved.')
        # early stopping
        elif len(scores) - np.argmax(scores) > 4:
            print('stopping training.')
            break
    return model        

def main(args):
    global encoderTokenizer, decoderTokenizer, DEVICE

    print('using device:', DEVICE)
    print('save path:', args.save_path)
    print('model:', args.model)

    STEPS = 138500 // args.batch_size                                           # total training samples / batch size
    encoderTokenizer, decoderTokenizer, model = select_model(mod=args.model)

    model.config.max_length = args.max_length
    model.config.no_repeat_ngram_size = 3
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=args.lr*10,
    #     steps_per_epoch=STEPS,
    #     pct_start=0.15,
    #     epochs=args.epochs
    # )

    # start from last checkpoint
    if args.init_epoch > 0:
        print('loading from', args.save_path)
        checkpoint = torch.load(args.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        INIT_EPOCH = checkpoint['epoch']
        
    model = train(model, optimizer, args)

    # torch.save(
    #     {
    #         'epoch': args.epochs,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #     }, args.save_path)
    # print('model saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training.')
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
        '--epochs', default=40, type=int,
        help='number of training epochs'
    )
    parser.add_argument(
        '--init_epoch', default=0, type=int,
        help='epoch to resume the training from'
    )
    parser.add_argument(
        '--batch_size', default=50, type=int,
        help='batch size for training'
    )
    parser.add_argument(
        '--lr', default=1e-5, type=float,
        help='learning rate for training'
    )
    parser.add_argument(
        '--save_path', default='../checkpoint/model.pt', type=str,
        help='model save path'
    )
    args = parser.parse_args()
    main(args)