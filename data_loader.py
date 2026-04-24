import numpy as np
import glob
import os
import torch
from tqdm import tqdm

ALL_FEATURES = ['mfcc', 'delta', 'delta2', 'zcr', 'rms']


def load_features(feature_files, flatten, return_gender, features=None):
    if features is None:
        features = ALL_FEATURES

    records = []

    for filepath in tqdm(feature_files, desc="Loading data"):
        filename = os.path.basename(filepath)
        parts = filename.replace('.npz', '').split('-')

        emotion_code = parts[2]
        actor_id = int(parts[6])
        is_augmented = int(parts[7])
        data = np.load(filepath)

        record = {
            'label': int(emotion_code) - 1,
            'actor_id': actor_id,
            'is_augmented': is_augmented # 1 is augmented, 0 is not
        }

        if return_gender:
            record['gender'] = actor_id % 2

        for feat in features:
            record[feat] = data[feat]

        records.append(record)

    return records


def fit_normalizer(X):
    return X.mean(axis=0), X.std(axis=0)


def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def pad_or_truncate(x, max_len):
    W = x.shape[-1]
    if W >= max_len:
        return x[..., :max_len]
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, max_len - W)]
    return np.pad(x, pad_width)


def apply_cmvn(records, feature_keys, train_actor_ids):
    for key in feature_keys:
        actor_frames = {}
        for r in records:
            if r['actor_id'] not in train_actor_ids:
                continue
            aid = r['actor_id']
            if aid not in actor_frames:
                actor_frames[aid] = []
            actor_frames[aid].append(r[key])

        actor_stats = {}
        for aid, clips in actor_frames.items():
            all_frames = np.concatenate(clips, axis=-1)
            actor_stats[aid] = {
                'mean': all_frames.mean(axis=-1, keepdims=True),
                'std':  all_frames.std(axis=-1, keepdims=True)
            }

        global_mean = np.mean([s['mean'] for s in actor_stats.values()], axis=0)
        global_std  = np.mean([s['std']  for s in actor_stats.values()], axis=0)

        for r in records:
            aid = r['actor_id']
            mean = actor_stats[aid]['mean'] if aid in actor_stats else global_mean
            std  = actor_stats[aid]['std']  if aid in actor_stats else global_std
            r[key] = (r[key] - mean) / (std + 1e-8)

    return records


def prepare_data(feature_files, flatten=True, features=None, return_gender=False,
                 test_actors=None, val_per_gender=2, seed=6283, max_len=None,
                 apply_cmvn_flag=False):

    if not flatten and max_len is None:
        raise ValueError("Set max_len before calling prepare_data with flatten=False.")

    if test_actors is None:
        test_actors = {21, 22, 23, 24}

    if features is None:
        features = ALL_FEATURES

    records = load_features(feature_files, flatten=False,
                            return_gender=return_gender, features=features)

    # compute splits before CMVN so train_actors is known
    uniq_actor_id = np.unique([r['actor_id'] for r in records
                                if r['actor_id'] not in test_actors])
    np.random.seed(seed)

    male_set   = [a for a in uniq_actor_id if a % 2 == 1]
    female_set = [a for a in uniq_actor_id if a % 2 == 0]

    np.random.shuffle(male_set)
    np.random.shuffle(female_set)

    val_actors   = set(male_set[:val_per_gender] + female_set[:val_per_gender])
    train_actors = set(a for a in uniq_actor_id if a not in val_actors)

    # apply CMVN per feature before anything else
    if apply_cmvn_flag:
        records = apply_cmvn(records, features, train_actors)

    # pad or truncate each feature to max_len, then concatenate into single vector
    for r in records:
        arrays = []
        for feat in features:
            arr = r[feat]
            if not flatten:
                arr = pad_or_truncate(arr, max_len)
            else:
                # mean over time axis for flat vector
                arr = arr.mean(axis=-1)
            arrays.append(arr)
        r['features'] = np.concatenate(arrays, axis=0)

    # split — val/test use only original (non-augmented) recordings
    test_data      = [r for r in records if r['actor_id'] in test_actors and r['is_augmented'] == 0]
    train_val_data = [r for r in records if r['actor_id'] not in test_actors]
    train_data     = [r for r in train_val_data if r['actor_id'] not in val_actors]
    val_data       = [r for r in train_val_data if r['actor_id'] in val_actors and r['is_augmented'] == 0]

    x_train = np.array([r['features'] for r in train_data])
    y_train = np.array([r['label']    for r in train_data])
    x_val   = np.array([r['features'] for r in val_data])
    y_val   = np.array([r['label']    for r in val_data])
    x_test  = np.array([r['features'] for r in test_data])
    y_test  = np.array([r['label']    for r in test_data])

    # global normalization only when CMVN is not used
    if not apply_cmvn_flag:
        train_mean, train_std = fit_normalizer(x_train)
        x_train = normalize(x_train, train_mean, train_std)
        x_val   = normalize(x_val,   train_mean, train_std)
        x_test  = normalize(x_test,  train_mean, train_std)

    x_train_torch = torch.tensor(x_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    x_val_torch   = torch.tensor(x_val,   dtype=torch.float32)
    y_val_torch   = torch.tensor(y_val,   dtype=torch.long)
    x_test_torch  = torch.tensor(x_test,  dtype=torch.float32)
    y_test_torch  = torch.tensor(y_test,  dtype=torch.long)

    return x_train_torch, y_train_torch, x_val_torch, y_val_torch, x_test_torch, y_test_torch