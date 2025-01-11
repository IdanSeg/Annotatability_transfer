
from dataset import Dataset

class Minimal(Dataset):
    def load_data(self):
        # Load data and save it as an instance attribute
        self.data = "/cs/usr/idan724/lab/merlin_cxg_minimal/merlin_cxg_2023_05_15_sf-log1p_minimal"
        return self.data

    def preprocess_data(self):
        return self.data
