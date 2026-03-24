import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


class HiRISEDataset(Dataset):
    """
    A custom Dataset for loading images.
    A dir contains several sets of images.
    Eg:
    - npy_files/                                    # Root directory
        - ESP_026721_1635/                          # Observation directory
            - ESP_026721_1635_0_RED5.npy            # Image array - shape either (32, 1024) or (256, 256)
            - ESP_026721_1635_0_RED4.npy
            - ESP_026721_1635_0_RED3.npy
            - ESP_026721_1635_0_IR10.npy
            - ESP_026721_1635_0_BG12.npy

            - ESP_026721_1635_1_RED5.npy
            - ESP_026721_1635_1_RED4.npy
            - ESP_026721_1635_1_RED3.npy
            - ESP_026721_1635_1_IR10.npy
            - ESP_026721_1635_1_BG12.npy

            - etc

        - ESP_011290_1800/
            - etc

    Args:
        data_record (pd.DataFrame): Data record containing the paths to the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_record, data_root=None, transform=None, meta_cols=None, scale_pix=False, norm_mode="scene_robust", global_center=None, global_scale=None, pix_min=None, pix_max=None, meta_scaler=None):
        self.transform = transform              # Image transformations
        self.data_record = data_record          # Data record
        self.meta_cols = meta_cols              # Meta values (list of column names)
        # data_root is prepended to every row['Path'] from the CSV.
        # Falls back to <cwd>/data to preserve the original behaviour.
        self.data_root = data_root if data_root else os.path.join(os.getcwd(), "data")
        self.unique_sets = sorted(self.data_record['Set'].unique())
        self.scale_pix = scale_pix              # Whether to scale BG&IR to RED mean
        self.norm_mode = norm_mode  # "scene_robust" | "global_robust" | None
        self.global_center = global_center
        self.global_scale = global_scale
        self.band_ccds = ['IR10', 'BG12']
        self.neighbour_ccds = ['RED5', 'RED3']

        if self.meta_cols is not None:
            missing = [c for c in self.meta_cols if c not in self.data_record.columns]
            assert not missing, f"Meta columns {missing} not found in data record"
            if meta_scaler is None:
                self.meta_scaler = StandardScaler().fit(self.data_record[self.meta_cols])
            else:
                self.meta_scaler = meta_scaler

        if pix_min is None or pix_max is None:
            pix_max = self.data_record['Pix_max'].max()
            pix_min = self.data_record['Pix_min'].min()
        self.pix_max = pix_max
        self.pix_min = pix_min
        self.denom = (pix_max - pix_min) + 1e-8

        assert pix_max <= 1 and pix_max >= 0, f"Pixel maximum value should be in the range [0, 1] but found {pix_max}"
        assert pix_min < pix_max and pix_min >= -1, (
            f"Pixel minimum should be less than pixel max and greater than -1 but found {pix_min} (max: {pix_max})")
        assert norm_mode in ['scene_robust', 'global_robust', None], f"Invalid norm_mode given: {norm_mode}"

        rand_line = self.data_record.sample(1, random_state=42).iloc[0]
        try:
            img = np.load(rand_line['Path'])
            assert img.dtype == np.float32, f"Image should be of type float32, not {img.dtype}"
        except FileNotFoundError:
            pass

        # Confirm every path in dr exists
        invalid_obs = []

        for idx, row in self.data_record.iterrows():
            path = os.path.join(self.data_root, row['Path'])
            if self.data_root[0] in ("D", "d"):  # Windows path fix
                path = path.replace("/", "\\")
            if not os.path.isfile(path):
                observation = row['Observation']
                invalid_obs.append(observation)

        if invalid_obs:
            original_count = self.data_record['Observation'].nunique()
            # Remove whole obs from dr
            invalid_obs = set(invalid_obs)
            self.data_record = self.data_record[~self.data_record['Observation'].isin(invalid_obs)]
            raise ValueError(f"WARNING: {len(invalid_obs)} observations out of {original_count} were removed from the dataset due to missing files\n"
                  f"Example of invalid path: {path}")

    def __len__(self):
        return self.data_record['Set'].nunique()

    def __getitem__(self, idx):
        set_idx = self.unique_sets[idx]
        df = self.data_record[self.data_record['Set'] == set_idx].set_index('CCD')

        # ------ LOAD IMAGES ------
        def load(rel_path: str):
            return np.load(os.path.join(self.data_root, rel_path))

        try:
            band_imgs = [load(df.at[ccd, 'Path']) for ccd in self.band_ccds]
            neighbour_imgs = [load(df.at[ccd, 'Path']) for ccd in self.neighbour_ccds]
            label_img = load(df.at['RED4', 'Path'])
        except FileNotFoundError as e:
            # print("df:\n", df, flush=True)
            raise FileNotFoundError(f"(Inside __getitem__) File not found for set {set_idx}: {e}")

        assert band_imgs[0].dtype == np.float32, f"Band images should be of type float32, not {band_imgs[0].dtype}"
        assert neighbour_imgs[0].dtype == np.float32, f"Neighbour images should be of type float32, not {neighbour_imgs[0].dtype}"
        assert label_img.dtype == np.float32, f"Label image should be of type float32, not {label_img.dtype}"

        # ------ LOAD META VALUES ------
        if self.meta_cols is not None and len(self.meta_cols) > 0:
            # !!!! CAUTION - Binning and TDI vary by CCD !!!
            raw_meta = pd.DataFrame([df.loc['RED3', self.meta_cols]], columns=self.meta_cols)    # (1, n_meta)
            meta_vals = self.meta_scaler.transform(raw_meta).squeeze(0)
            meta_tensor = torch.tensor(meta_vals, dtype=torch.float32)                # (n_meta,)
        else:
            meta_tensor = torch.empty(0, dtype=torch.float32)

        # ------ BUILD SAMPLE  ------
        obs_id = df.at['IR10', 'Observation']
        set_name = df.at['IR10', 'Set']
        date = df.at['IR10', 'Date']

        bands = torch.from_numpy(np.stack(band_imgs)).float()               # bands: (2, Hb, Wb)
        neighbours = torch.from_numpy(np.stack(neighbour_imgs)).float()     # neighbours: (2, Hn, Wn, 4)
        label = torch.from_numpy(label_img).float().unsqueeze(0)            # label: (1, Hb, Wb)

        # choose a shared affine (a,s)
        if self.norm_mode == "scene_robust":
            a, s = _robust_center_scale_from_inputs(bands, neighbours, include_red_means=False)
        elif self.norm_mode == "global_robust":
            assert self.global_center is not None and self.global_scale is not None, "global stats required"
            a = torch.tensor(float(self.global_center), dtype=bands.dtype)
            s = torch.tensor(float(self.global_scale), dtype=bands.dtype)
        else:
            # norm_mode not recognised, defaulting to a=0.0, s=1.0
            a = torch.tensor(0.0, dtype=bands.dtype)
            s = torch.tensor(1.0, dtype=bands.dtype)

        # apply the SAME affine to inputs and label (train-time)
        bands = (bands - a) / s
        neighbours = (neighbours - a) / s
        label = (label - a) / s

        x_band = bands                # (2, Hb, Wb)
        neigh = neighbours[..., :3].permute(0, 3, 1, 2)    # (2, 3, Hn, Wn) - Drop r4
        x_neigh = neigh.reshape(-1, neigh.shape[-2], neigh.shape[-1])    # (6, Hn, Wn)

        if meta_tensor.numel() > 0:
            x_meta = meta_tensor.to(dtype=bands.dtype)  # (n_meta,)
        else:
            x_meta = torch.empty(0, dtype=bands.dtype)

        y = label

        # Capture stats for denormalization
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=bands.dtype)
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=bands.dtype)
        stats = torch.stack([a, s])

        chan_spec_band = ['IR10', 'BG12']
        chan_spec_neigh = [
            'R5_BG',
            'R5_IR',
            'R5_RE',
            'R3_BG',
            'R3_IR',
            'R3_RE',
        ]

        if self.meta_cols is not None and len(self.meta_cols) > 0:
            meta_spec = [f'META:{c}' for c in self.meta_cols]
        else:
            meta_spec = []

        sample = dict(
            x_band=x_band,                  # (2, Hb, Wb)
            x_neigh=x_neigh,                # (6, Hn, Wn)
            x_meta=x_meta,                  # (n_meta,)
            y=y,                            # (1, Hb, Wb)
            stats=stats,
            chan_spec_band=chan_spec_band,
            chan_spec_neigh=chan_spec_neigh,
            meta_spec=meta_spec,
            obs_id=obs_id,
            set_name=set_name,
            date=date,
        )

        assert x_band.shape[0] == len(chan_spec_band), (f"bands channels mismatch: {x_band.shape[0]=} vs {len(chan_spec_band)=}")
        assert x_neigh.shape[0] == len(chan_spec_neigh), (f"neigh channels mismatch: {x_neigh.shape[0]=} vs {len(chan_spec_neigh)=}")
        assert x_meta.numel() == len(meta_spec) or x_meta.numel() == 0, (f"meta length mismatch: {x_meta.numel()=} vs {len(meta_spec)=}")

        if self.transform:
            sample = self.transform(sample)
        return sample


class FilteredHiRISEDataset(HiRISEDataset):
    def __init__(self, data_record, sweep, data_root=None, transform=None, allowed_sets=None, meta_cols=None, norm_mode="scene_robust", global_center=None, global_scale=None, pix_min=None, pix_max=None, meta_scaler=None):
        # Filter data record to only include allowed sets
        if allowed_sets is not None:
            set_orig = data_record['Set'].nunique()
            data_record = data_record[data_record['Set'].isin(allowed_sets)]
            if data_record.empty:
                raise ValueError("No data found for the specified allowed sets. None of the requested sets are present.")
            if not sweep:
                print("Filtered dataset:")
                print(f"\tObservations: {data_record['Observation'].nunique():,}")
                print(f"\tSets: {data_record['Set'].nunique():,}")
                print(f"\tImages: {len(data_record):,}")
                print(f"\tProportion of sets: {data_record['Set'].nunique() / set_orig:.2%}")
                print()

        # Call the parent constructor with updated data record
        super().__init__(data_record, data_root=data_root, transform=transform, meta_cols=meta_cols, pix_min=pix_min, pix_max=pix_max, meta_scaler=meta_scaler, norm_mode=norm_mode, global_center=global_center, global_scale=global_scale)


class DiffusionDataset(FilteredHiRISEDataset):
    """
    Dataset for CM-Diff-style bidirectional diffusion training.

    Each sample exposes two modalities for BDT:
        ir   : IR10 band  (1, H, W)  — near-infrared
        red  : RED4 band  (1, H, W)  — panchromatic target

    Spatial neighbours (RED3, RED5) are kept as prior context for SCI
    inference (statistical / histogram constraints) but are NOT fed into
    the UNet during training.

    Returned keys
    -------------
    ir          : (1, H, W)  IR10, normalised
    red         : (1, H, W)  RED4, normalised
    prior       : (2, H', W')  RED3 + RED5, same normalisation — SCI prior
    norm_stats  : (2,)       [center, scale] used for this scene
    obs_id      : str
    set_name    : int
    date        : str
    """

    def __getitem__(self, idx):
        raw = super().__getitem__(idx)

        # x_band = (2, H, W): [IR10, BG12]
        # x_neigh = (6, H, W): [R5_BG, R5_IR, R5_RE, R3_BG, R3_IR, R3_RE]
        # y = (1, H, W): RED4

        ir  = raw['x_band'][[0]]        # (1, H, W)  IR10 only, drop BG12
        red = raw['y']                  # (1, H, W)  RED4

        # RED5_IR = channel 1,  RED3_IR = channel 4  (same band as IR10)
        red5 = raw['x_neigh'][[1]]      # (1, H, W)
        red3 = raw['x_neigh'][[4]]      # (1, H, W)
        prior = torch.cat([red5, red3], dim=0)   # (2, H, W)

        return dict(
            ir=ir,
            red=red,
            prior=prior,
            norm_stats=raw['stats'],
            obs_id=raw['obs_id'],
            set_name=raw['set_name'],
            date=raw['date'],
        )


def diffusion_collate_fn(batch):
    return dict(
        ir        =torch.stack([b['ir']         for b in batch]),   # (B, 1, H, W)
        red       =torch.stack([b['red']        for b in batch]),   # (B, 1, H, W)
        prior     =torch.stack([b['prior']      for b in batch]),   # (B, 2, H, W)
        norm_stats=torch.stack([b['norm_stats'] for b in batch]),   # (B, 2)
        obs_id    =[b['obs_id']   for b in batch],
        set_name  =[b['set_name'] for b in batch],
        date      =[b['date']     for b in batch],
    )


def collate_fn(batch):
    # x = torch.stack([b['x'] for b in batch], dim=0)    # (B, C_in, H, W)
    x_band = torch.stack([b['x_band'] for b in batch], dim=0)    # (B, 2, Hb, Wb)
    x_neigh = torch.stack([b['x_neigh'] for b in batch], dim=0)  # (B, 6, Hn, Wn)
    x_meta = torch.stack([b['x_meta'] for b in batch], dim=0)      # (B, n_meta) or (B, 0)
    y = torch.stack([b['y'] for b in batch], dim=0)    # (B, 1, H, W)
    # specs = [b['chan_spec'] for b in batch]
    stats = torch.stack([b['stats'] for b in batch], dim=0)
    cspecs_band = [b['chan_spec_band'] for b in batch]
    cspecs_neigh = [b['chan_spec_neigh'] for b in batch]
    cspecs_meta = [b['meta_spec'] for b in batch]
    # assert all(s == specs[0] for s in specs), "Channel specification mismatch across batch samples"
    return dict(
        x_band=x_band,
        x_neigh=x_neigh,
        x_meta=x_meta,
        y=y,
        stats=stats,
        chan_spec_band=cspecs_band[0],
        chan_spec_neigh=cspecs_neigh[0],
        meta_spec=cspecs_meta[0],
        obs_id=[b['obs_id'] for b in batch],
        set_name=[b['set_name'] for b in batch],
        date=[b['date'] for b in batch],
    )


def get_loader(dataset, batch_size, shuffle=True, pin_memory=True, prefetch_factor=2,
               num_workers=4, collate_fn=None, worker_init_fn=None, persistent_workers=True, seed=42):
    torch.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )


def _robust_center_scale_from_inputs(bands, neighbours=None, include_red_means=False, eps=1e-8):
    # bands: (2,H,W) -> IR, BG
    ir = bands[0].reshape(-1)
    bg = bands[1].reshape(-1)
    flat = torch.cat([ir, bg], dim=0)  # only pixel-aligned fields

    if include_red_means and neighbours is not None:
        # R5/R3 exist but are not pixel-aligned; use their per-patch means as *scalars*
        # neighbours shape ~ (2, H, Wn, 4), with RED at [..., 2]
        r5_mean = neighbours[0, :, :, 2].mean().reshape(1).to(flat.dtype)
        r3_mean = neighbours[1, :, :, 2].mean().reshape(1).to(flat.dtype)
        flat = torch.cat([flat, r3_mean, r5_mean], dim=0)

    a = flat.median()
    mad = (flat - a).abs().median()
    s = (1.4826 * mad).clamp_min(1e-3)
    return a, s


# Test usage
if __name__ == '__main__':
    start_time = time.time()
    transform = None

    project_root = os.getcwd()

    dr = pd.read_csv(project_root + '/data/files/data_record_bin12.csv')

    allowed_sets = [19645, 7292, 7293, 14774]  # observations present in data/files/npy_files_b12

    dataset = FilteredHiRISEDataset(transform=transform, data_record=dr, meta_cols=['Binning', 'TDI'], sweep=False, allowed_sets=allowed_sets)
    print(f"Dataset initialisation time: {time.time() - start_time:.2f} seconds")

    sample = dataset[0]

    print("Test usage:")
    print(f"\tDataset length: {len(dataset)}")
    print(f"\tNumber of neighbour images: {len(sample['x_neigh'])}")
    for img in sample['x_neigh']:
        print(f"\tImage shape: {img.shape}")
    print(f"\tLabel image shape: {sample['y'].shape}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

    # Inspect y per set directly from the dataset (bypasses DataLoader/collate)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        y = sample["y"]                 # (1, Hb, Wb), already a torch.Tensor
        set_name = sample["set_name"]   # whatever you stored in __getitem__

        y_flat = y.view(-1)
        y_mean = y_flat.mean().item()
        y_std = y_flat.std().item()

        print(f"Set {set_name}: mean={y_mean:.6f}, std={y_std:.6f}")

    loader = get_loader(dataset, batch_size=32, collate_fn=collate_fn)
    first_batch = next(iter(loader))

    y_batch = first_batch["y"].detach().cpu()  # (B, 1, Hb, Wb)
    B = y_batch.shape[0]
    per_img_std = y_batch.view(B, -1).std(dim=1)
    avg_std = per_img_std.mean().item()
    print(f"Avg per-image std of first batch (y): {avg_std:.6f}")
