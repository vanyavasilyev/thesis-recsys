from PIL import Image
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, data_dir, triplets_df, preprocessor, full_paths=False):
        self.data_dir = data_dir
        self.triplets = triplets_df
        self.preprocessor = preprocessor
        self.full_paths = full_paths

    def __len__(self):
        return len(self.triplets)

    def _load_img(self, path):
        if not self.full_paths:
            img = Image.open(f"{self.data_dir}/{path}")
        else:
            img = Image.open(path)
        return self.preprocessor(img.convert('RGB'))

    def __getitem__(self, idx):
        paths = self.triplets.iloc[idx]
        return tuple(self._load_img(p) for p in paths)