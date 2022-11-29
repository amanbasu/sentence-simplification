import sari
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data import DataLoader
from datagen import DataGenerator

def get_dataloader(batch_size):
    trainDataset = DataGenerator(mode='train')
    valDataset = DataGenerator(mode='valid')

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    valLoader = DataLoader(
        valDataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )
    return iter(trainLoader), iter(valLoader)

def encode_batch(tokenizer, src, tgt, max_len=100):
    src_tok = tokenizer(
        src, 
        max_length=max_len, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

    tgt_tok = tokenizer(
        tgt, 
        max_length=max_len, 
        padding='max_length',
        truncation=True, 
        return_tensors='pt'
    )
    
    return src_tok.input_ids, tgt_tok.input_ids

def bleu_score(pred, ref):
    size = len(pred)
    weights = (0.25, 0.25, 0.25, 0.25)
    chencherry = SmoothingFunction().method1
    
    score = 0
    for a, b in zip(ref, pred):
        for idx, sent in enumerate(a):
            a[idx] = sent.split(' ')
        b = b.split(' ')
        score += corpus_bleu([a], [b], weights=weights, smoothing_function=chencherry)

    return score / size

def sari_score(src, pred, ref):
    size = len(pred)
    score = 0
    for a, b, c in zip(src, pred, ref):
        score += sari.SARIsent(a, b, c)
    return score / size

