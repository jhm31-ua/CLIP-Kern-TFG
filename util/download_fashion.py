import os
import shutil
from torchvision import datasets, transforms
from PIL import Image

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

directory = 'my-datasets/fashion-dataset'
os.makedirs(directory, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.FashionMNIST(root='./temp', train=True, download=True, transform=transform)

for idx in range(len(dataset)):
    image, label = dataset[idx]
    
    image_pil = transforms.ToPILImage()(image)
    image_path = os.path.join(directory, f'{idx}.png')
    image_pil.save(image_path)
    
    label_text = class_names[label]  
    label_path = os.path.join(directory, f'{idx}.krn')
    
    with open(label_path, 'w') as label_file:
        label_file.write(label_text) 

shutil.rmtree('./temp')
