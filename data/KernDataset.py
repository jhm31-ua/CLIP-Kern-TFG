import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

from model.BPEKernTokenizer import BPEKernTokenizer
from util import Parameters

class KernDataset(Dataset):
    def __init__(self, set_name, k, split_path = Parameters.split_path, dataset_path = Parameters.dataset_path, tokenizer_path = Parameters.tokenizer_path, max_seq_length = Parameters.max_seq_length): 
        self.dataset_path = dataset_path
        self.max_seq_length = max_seq_length

        self.data_files = open(f"{split_path}/{k}/{set_name}.dat", 'r').read().split("\n")
        self.tokenizer = BPEKernTokenizer(filepath = tokenizer_path, load = True)

    def transform(self, img):
        img = T.Resize(Parameters.img_size)(img)
        img = T.ToTensor()(img)
        #img = 1.0 - img # Inverted colours

        return img
        
    def __len__(self):
        return len(self.data_files) 

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        i_path = os.path.join(self.dataset_path, f"{data_file}.png")
        c_path = os.path.join(self.dataset_path, f"{data_file}.krn")

        img = Image.open(i_path).convert("L")
        img = self.transform(img)

        with open(c_path, 'r') as f:
            cap = f.read().strip()

        cap, mask = self.tokenizer.encode(cap, max_seq_length = self.max_seq_length)
        mask = mask.repeat(len(mask), 1) # Se añade otra dimensión para el mecanismo de atención

        return {"image": img, "caption": cap, "mask": mask}

