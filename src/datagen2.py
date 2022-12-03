from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, dir='../dataset'):
        self.dir = dir
        self.oscar = self.read_data('oscar.en.txt')

    def __len__(self):
        return len(self.oscar)

    def __getitem__(self, idx):       
        return self.oscar[idx]

    def read_data(self, file):        
        with open(f'{self.dir}/{file}', 'r') as f:
            return f.readlines()