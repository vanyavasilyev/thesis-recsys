import tqdm
import torch

from PIL import Image
from torch.utils.data import Dataset


class CategoryDataset(Dataset):
    def __init__(self, category_df, base_categories, category_mapping, preprocessor):
        self.df = category_df
        self.preprocessor = preprocessor
        self.gt_probabilities = torch.zeros((len(category_df), len(base_categories)))
        for i, name in enumerate(tqdm.tqdm(self.df.category_name)):
            current_base_categories = category_mapping[name]
            for cat in current_base_categories:
                self.gt_probabilities[i,base_categories[cat]] = 1 / len(current_base_categories)

    def __len__(self):
        return len(self.df)

    def _load_img(self, path):
        img = Image.open(path)
        return self.preprocessor(img)

    def __getitem__(self, idx):
        path = self.df.filenames.iloc[idx]
        img = self._load_img(path)
        probs = self.gt_probabilities[idx,:]
        return (img, probs)
