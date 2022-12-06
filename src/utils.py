import sari
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from easse.fkgl import corpus_fkgl
from easse.sari import corpus_sari
from torch.utils.data import DataLoader
from datagen import DataGenerator
from transformers import GPT2TokenizerFast, GPT2Config, EncoderDecoderConfig
from transformers import BertTokenizerFast, BertConfig, EncoderDecoderModel

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

def get_testloader(batch_size):
    testDataset = DataGenerator(mode='test')
    testLoader = DataLoader(
        testDataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    return iter(testLoader)

def encode_batch(encoderTokenizer, decoderTokenizer, src, tgt, max_len=100):
    src_tok = encoderTokenizer(
        src, 
        max_length=max_len, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

    tgt_tok = decoderTokenizer(
        tgt, 
        max_length=max_len, 
        padding='max_length',
        truncation=True, 
        return_tensors='pt'
    )

    labels = tgt_tok.input_ids.clone()
    # labels[tgt_tok.attention_mask == 0] = -100
    
    return src_tok.input_ids, src_tok.attention_mask, tgt_tok.input_ids, tgt_tok.attention_mask, labels

def bleu_score(pred, ref):
    weights = (0.25, 0.25, 0.25, 0.25)
    chencherry = SmoothingFunction().method1
    
    scores = []
    for a, b in zip(ref, pred):
        for idx, sent in enumerate(a):
            a[idx] = sent.split(' ')
        b = b.split(' ')
        scores += [corpus_bleu([a], [b], weights=weights, smoothing_function=chencherry)]

    return scores

def sari_score(src, pred, ref):
    scores = []
    for a, b, c in zip(src, pred, ref):
        # scores += [sari.SARIsent(a, b, c)]
        c = [[k] for k in c]
        scores += [corpus_sari([a], [b], c)]
    return scores

def fkgl_score(pred):
    scores = []
    for a in pred:
        scores += [corpus_fkgl([a])]
    return scores
    
def select_model(mod='gpt2'):

    encoderConfig = decoderConfig = None

    if mod == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        encoderTokenizer = decoderTokenizer = tokenizer

        encoderConfig = decoderConfig = GPT2Config()

    elif mod == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token

        encoderTokenizer = decoderTokenizer = tokenizer

        encoderConfig = decoderConfig = BertConfig()

    elif mod == 'bert_gpt2':
        encoderTokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        encoderTokenizer.bos_token = encoderTokenizer.cls_token
        encoderTokenizer.eos_token = encoderTokenizer.sep_token

        decoderTokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        decoderTokenizer.pad_token = decoderTokenizer.eos_token

        encoderConfig = BertConfig()
        decoderConfig = GPT2Config()
        
    elif mod == 'gpt2_bert':
        encoderTokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        encoderTokenizer.pad_token = encoderTokenizer.eos_token

        decoderTokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        decoderTokenizer.bos_token = decoderTokenizer.cls_token
        decoderTokenizer.eos_token = decoderTokenizer.sep_token

        encoderConfig = GPT2Config()
        decoderConfig = BertConfig()

    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoderConfig, decoderConfig
    )

    model = EncoderDecoderModel(config=config)
    model.config.decoder_start_token_id = decoderTokenizer.bos_token_id
    model.config.eos_token_id = decoderTokenizer.eos_token_id
    model.config.pad_token_id = decoderTokenizer.pad_token_id
    
    return encoderTokenizer, decoderTokenizer, model
