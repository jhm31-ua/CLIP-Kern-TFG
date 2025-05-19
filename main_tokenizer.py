import fire

from util import Parameters
from model.BPEKernTokenizer import BPEKernTokenizer

class Main:
    def __init__(self,
                 tokenizer_path = Parameters.tokenizer_path,
                 dataset_path = Parameters.dataset_path,
                 vocab_size = Parameters.vocab_size):
        self.tokenizer_path = tokenizer_path
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size

    def train(self):
        BPEKernTokenizer(filepath = self.tokenizer_path, vocab_size = self.vocab_size, load = False).train_folder(self.dataset_path, verbose = True)

if __name__ == "__main__":
    fire.Fire(Main) # Ejemplo de uso: python main_tokenizer.py train --vocab_size=256