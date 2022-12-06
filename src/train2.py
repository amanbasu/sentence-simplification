from tqdm import tqdm
import torch
import numpy as np
import os
from utils import encode_batch, get_dataloader, sari_score, bleu_score, fkgl_score, select_model

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

mod = 'gpt2'
encoderTokenizer, decoderTokenizer = None, None
nlines = 20
STEPS = 2769
MAX_LENGTH = 80
BATCH_SIZE = 50
INIT_EPOCH = 0
EPOCHS = 50
LEARNING_RATE = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = f'../checkpoint/model_{mod}_v3.pt'

def train(model, optimizer, scheduler):

    scores = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        trainLoader, valLoader = get_dataloader(BATCH_SIZE)
        losses = []
        
        # when dataloader runs out of batches, it throws an exception
        try:
            for source, target in tqdm(trainLoader):
                src_inp, _, _, _, labels = encode_batch(
                    encoderTokenizer, decoderTokenizer, source, target
                )

                optimizer.zero_grad(set_to_none=True)                           # clear gradients w.r.t. parameters

                loss = model(
                    input_ids = src_inp.to(DEVICE), 
                    labels = labels.to(DEVICE)
                )[0]

                losses += [loss.item()]
                loss.backward()                                                 # getting gradients
                optimizer.step()                                                # updating parameters
                scheduler.step()   
        except StopIteration:
            pass

        # get model performace on val set
        with torch.no_grad():
            bleus, saris, fkgls = [], [], []
            try:
                for source, target, ref in tqdm(valLoader):

                    ref = np.array(ref).T.tolist()                              # transpose ref, order gets changed in datagen

                    src_inp, _, _, _, labels = encode_batch(
                        encoderTokenizer, decoderTokenizer, source, target
                    )

                    logits = model.generate(input_ids=src_inp.to(DEVICE), max_length=MAX_LENGTH)
                    outputs = decoderTokenizer.batch_decode(
                        logits, skip_special_tokens=True
                    )

                    sari = sari_score(source, outputs, ref)         
                    bleu = bleu_score(outputs, ref)
                    fkgl = fkgl_score(outputs)

                    saris += sari
                    bleus += bleu
                    fkgls += fkgl

            except StopIteration:
                pass

        losses = np.mean(losses)
        saris = np.mean(saris)
        bleus = np.mean(bleus)
        print(f'loss: {losses:.4f} - sari: {saris:.4f} - bleu: {bleus:.4f} - fkgl: {np.mean(fkgls):.4f}')
        # print(f"{idx//BATCH_SIZE+1}/{SIZE//BATCH_SIZE} [{'=' * progress}>{'-' * (nlines - progress)}] loss: {np.mean(losses):.3f}", end='\r')

        scores.append(saris)
        # save model checkpoint
        if epoch > 3 and scores[-1] == np.max(scores):
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, f'../checkpoint/model_{mod}_v3_best.pt')
            print('checkpoint saved.')
        elif len(scores) - np.argmax(scores) > 4:
            print('stopping training.')
            break

    return model        

if __name__ == '__main__':

    print('using device:', DEVICE)
    print('save path:', SAVE_PATH)
    print('model:', mod)

    encoderTokenizer, decoderTokenizer, model = select_model(mod=mod)

    model.config.max_length = MAX_LENGTH
    model.config.no_repeat_ngram_size = 3
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE*10,
        steps_per_epoch=STEPS,
        pct_start=0.15,
        epochs=EPOCHS
    )

    # start from last checkpoint
    if INIT_EPOCH > 0:
        print('loading from', SAVE_PATH)
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        INIT_EPOCH = checkpoint['epoch']
        
    model = train(model, optimizer, scheduler)

    torch.save(
        {
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVE_PATH)
    print('model saved.')