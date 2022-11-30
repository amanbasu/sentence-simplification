from tqdm import tqdm
import torch
import numpy as np
import os
from utils import encode_batch, get_dataloader, sari_score, bleu_score, select_model

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

mod = 'bert'
encoderTokenizer, decoderTokenizer = None, None
nlines = 20
MAX_LENGTH = 100
BATCH_SIZE = 50
INIT_EPOCH = 0
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = f'../checkpoint/model_{mod}.pt'
softmax = torch.nn.LogSoftmax(dim=-1)

def train(model, optimizer):

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        trainLoader, valLoader = get_dataloader(BATCH_SIZE)

        # when dataloader runs out of batches, it throws an exception
        try:
            for source, target in tqdm(trainLoader):
                src_inp, src_att, tgt_inp, tgt_att, labels = encode_batch(
                    encoderTokenizer, decoderTokenizer, source, target
                )

                optimizer.zero_grad(set_to_none=True)                           # clear gradients w.r.t. parameters

                loss = model(
                    input_ids = src_inp.to(DEVICE), 
                    decoder_input_ids = tgt_inp.to(DEVICE),
                    attention_mask = src_att.to(DEVICE),
                    decoder_attention_mask = tgt_att.to(DEVICE),
                    labels = labels.to(DEVICE)
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
                    outputs = decoderTokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )

                    sari = sari_score(source, outputs, ref)        
                    bleu = bleu_score(outputs, ref)

                    losses += [loss.item()]
                    saris += [sari]
                    bleus += [bleu]

            except StopIteration:
                pass

        losses = np.mean(losses)
        saris = np.mean(saris)
        bleus = np.mean(bleus)
        print(f'loss: {losses:.4f} - sari: {saris:.4f} - bleu: {bleus:.4f}')
        # print(f"{idx//BATCH_SIZE+1}/{SIZE//BATCH_SIZE} [{'=' * progress}>{'-' * (nlines - progress)}] loss: {np.mean(losses):.3f}", end='\r')

        # save model checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, f'../checkpoint/model_{mod}_{epoch+1}.pt')
        print('checkpoint saved.')

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