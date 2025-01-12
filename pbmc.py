
from dataset import Dataset
import scanpy as sc
import squidpy as sq
FILE_PATH = "/cs/labs/ravehb/idan724/annotatability/datasets/pbmc/dfb51f99-a306-4daa-9f4a-afe7de65bbf2.h5ad"

class PBMC(Dataset):
    def load_data(self):
        # Load data and save it as an instance attribute
        self.adata_pbmc = sc.read_h5ad(FILE_PATH)
        print(self.adata.obs.columns)  # List all metadata columns
        print(self.adata.obs.head())   # View the first few rows of metadata
        return self.adata_pbmc

    def preprocess_data(self):
        # Normalize and log-transform the data
        sc.pp.normalize_total(self.adata_pbmc, target_sum=1e4)
        sc.pp.log1p(self.adata_pbmc)
        return self.adata_pbmc

    def filter_by_health(self, clear_sick=True, normalize_again=False):
        if 'COVID_status' not in self.adata_pbmc.obs.columns:
            raise KeyError("'COVID_status' column not found in adata_full.obs.")

        if 'Healthy' not in self.adata_pbmc['COVID_status'].unique():
            raise ValueError("'Healthy' label not found in 'COVID_status' column.")

        filter_condition = self.adata_pbmc.obs['COVID_status'] == 'Healthy' if clear_sick else self.adata_pbmc.obs['COVID_status'] != 'Healthy'
        filtered_adata = self.adata_pbmc[filter_condition].copy()
        
        if filtered_adata.n_obs == 0:
            raise ValueError(f"No {'healthy' if clear_sick else 'sick'} cells found after filtering.")

        self.adata_pbmc = filtered_adata

        status = "healthy" if clear_sick else "sick"
        print(f"Filtered {status} cells using 'COVID_status'.")

        if normalize_again:
            return self.preprocess_data()
        
        return self.adata_pbmc