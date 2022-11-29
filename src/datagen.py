from torch.utils.data import Dataset
import pickle

class DataGenerator(Dataset):
    def __init__(self, mode='train', dir='../dataset'):
        self.dir = dir
        self.mode = mode
        self.src = self.read_data(f'src_{mode}.txt')
        self.tgt = self.read_data(f'tgt_{mode}.txt')

        if mode != 'train':
            self.pkl = self.read_data(f'ref_{mode}.pkl', pkl=True)

    def __len__(self):
        assert len(self.src) == len(self.tgt)
        return len(self.src)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.src[idx], self.tgt[idx]        
        return self.src[idx], self.tgt[idx], self.pkl[idx]

    def read_data(self, file, pkl=False):
        if pkl:
            return pickle.load(open(f'{self.dir}/{file}', 'rb'))
        
        with open(f'{self.dir}/{file}', 'r') as f:
            lines = f.readlines()
            return list(map(lambda x: x.strip(), lines))