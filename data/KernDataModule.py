import pytorch_lightning as pl
from torch.utils.data import DataLoader

from util import Parameters
from data.KernDataset import KernDataset

class KernDataModule(pl.LightningDataModule):
    def __init__(self, k, split_path = Parameters.split_path, dataset_path = Parameters.dataset_path, tokenizer_path = Parameters.tokenizer_path, max_seq_length = Parameters.max_seq_length, batch_size = Parameters.batch_size):
        super().__init__()

        self.k = k
        self.split_path = split_path
        self.dataset_path = dataset_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.train_set = KernDataset('train', self.k, self.split_path, self.dataset_path, self.tokenizer_path, max_seq_length = self.max_seq_length)
            self.validation_set = KernDataset('validation', self.k, self.split_path, self.dataset_path, self.tokenizer_path, max_seq_length = self.max_seq_length) 
       
        if stage == "test" or stage is None:
             self.test_set = KernDataset('test', self.k, self.split_path, self.dataset_path, self.tokenizer_path, max_seq_length = self.max_seq_length) 
    
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle = False, batch_size = self.batch_size, drop_last = False)

    def val_dataloader(self):
        return DataLoader(self.validation_set, shuffle = False, batch_size = self.batch_size, drop_last = False)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle = False, batch_size = self.batch_size, drop_last = False)