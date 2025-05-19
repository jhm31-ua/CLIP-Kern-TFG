import numpy as np
import os

class DataSplit:
    def __init__(self, dataset_path, k_folds):
        self.dataset_path = dataset_path
        self.dataset = os.path.basename(os.path.normpath(self.dataset_path))
        self.k_folds = k_folds
    
    def k_fold_split(self):
        base_folder = f'splits/{self.dataset}/k{self.k_folds}'
        os.makedirs(base_folder, exist_ok = True)

        f = sorted([i[:-4] for i in os.listdir(self.dataset_path) if i.endswith('.png')])

        fold_size = len(f) // self.k_folds
        folds = [f[i * fold_size : (i + 1) * fold_size] for i in range(self.k_folds)]
        
        remaining_files = f[self.k_folds * fold_size:]
        for idx, file in enumerate(remaining_files):
            folds[idx % self.k_folds].append(file)

        for i in range(self.k_folds):
            test = folds[i]
            validation = folds[(i + 1) % self.k_folds]
            train = [item for j, fold in enumerate(folds) if j != i and j != (i + 1) % self.k_folds for item in fold]

            fold_folder = os.path.join(base_folder, str(i))
            os.makedirs(fold_folder, exist_ok = True)

            with open(os.path.join(fold_folder, 'train.dat'), 'w') as f_train:
                f_train.write('\n'.join(train))
            with open(os.path.join(fold_folder, 'validation.dat'), 'w') as f_validation:
                f_validation.write('\n'.join(validation))
            with open(os.path.join(fold_folder, 'test.dat'), 'w') as f_test:
                f_test.write('\n'.join(test))

DataSplit('my-datasets/HOMUS-parsed', 5).k_fold_split()

        
