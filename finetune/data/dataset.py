import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..config import BaseConfig


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train', config=None, suffix: str = "_real"):
        self.config = config if config is not None else BaseConfig()
        if data_type not in ['train', 'val', 'test']:
            raise ValueError("data_type must be 'train', 'val', or 'test'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data{suffix}.pkl"
            self.n_samples = self.config.n_train_iter
        elif data_type == 'val':
            self.data_path = f"{self.config.dataset_path}/val_data{suffix}.pkl"
            self.n_samples = self.config.n_val_iter
        else:  # test
            self.data_path = f"{self.config.dataset_path}/test_data{suffix}.pkl"
            self.n_samples = getattr(self.config, 'n_test_iter', 1000)  # Default test samples

        try:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_path}\n"
                f"Please run data preparation first: python run_complete_finetune.py --config {data_type}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {self.data_path}: {e}")

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].copy()
            
            # Reset index to ensure we have a proper integer index
            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()
            
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features if datetime column exists
                if 'datetime' in df.columns:
                    try:
                        df['minute'] = pd.to_datetime(df['datetime']).dt.minute
                        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
                        df['weekday'] = pd.to_datetime(df['datetime']).dt.weekday
                        df['day'] = pd.to_datetime(df['datetime']).dt.day
                        df['month'] = pd.to_datetime(df['datetime']).dt.month
                    except Exception as e:
                        print(f"Warning: Failed to generate time features for {symbol}: {e}")
                        # Use dummy time features
                        for feature in self.time_feature_list:
                            df[feature] = 0
                elif hasattr(df.index, 'minute'):  # DatetimeIndex
                    try:
                        df['minute'] = df.index.minute
                        df['hour'] = df.index.hour
                        df['weekday'] = df.index.weekday
                        df['day'] = df.index.day
                        df['month'] = df.index.month
                    except Exception as e:
                        print(f"Warning: Failed to extract time features from index for {symbol}: {e}")
                        # Use dummy time features
                        for feature in self.time_feature_list:
                            df[feature] = 0
                else:
                    # No datetime information available, use dummy features
                    for feature in self.time_feature_list:
                        df[feature] = 0
                
                # Keep only necessary columns to save memory
                available_features = [f for f in self.feature_list if f in df.columns]
                available_time_features = [f for f in self.time_feature_list if f in df.columns]
                
                if len(available_features) == 0:
                    raise ValueError(f"No required features found in data for {symbol}. "
                                   f"Required: {self.feature_list}, Available: {list(df.columns)}")
                
                self.data[symbol] = df[available_features + available_time_features]

                # Add all valid starting indices for this symbol to the global list
                for i in range(num_samples):
                    self.indices.append((symbol, i))
            else:
                print(f"Warning: Symbol {symbol} has insufficient data ({series_len} samples, need {self.window})")

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`. This
        ensures random sampling over the entire dataset for each call.

        Args:
            idx (int): Ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
                - x_stamp_tensor (torch.Tensor): The time feature tensor.
        """
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Perform instance-level normalization.
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")
