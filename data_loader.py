import numpy as np
import glob
import os
from tqdm import tqdm
# Insert to loop


def load_features(feature_files,flatten,return_gender):

    records = []

    for filepath in tqdm(feature_files, desc="Loading data"):
        filename = os.path.basename(filepath)
        parts = filename.replace('.npz', '').split('-')

        emotion_code = parts[2] # 01-08 (see EMOTION_MAP)

        actor_id = int(parts[6]) # 01-24

        data = np.load(filepath)

        # Feature vector building
        if flatten:
            # Mean over windows of features (We classify full samples)
            mfcc = data['mfcc'].mean(axis=1)      #(40,)
            delta = data['delta'].mean(axis=1)    # (40,)
            delta2 = data['delta2'].mean(axis=1)  # (40,)
            zcr = data['zcr'].mean(axis=1)        # (1,)
            rms = data['rms'].mean(axis=1)        # (1,)

        else:
            # Features for each window (keep temporal info)
            mfcc = data['mfcc']      # (40, W)
            delta = data['delta']    # (40, W)
            delta2 = data['delta2']  # (40, W)
            zcr = data['zcr']        # (1,  W)
            rms = data['rms']        # (1,  W)

        feature_vector = np.concatenate([mfcc, delta, delta2, zcr, rms], axis=0)

        if return_gender:
            records.append({
                'features':feature_vector,
                'label':int(emotion_code)-1, # CrossEntropyLoss (pytorch) expects [0-N-1] labels
                'actor_id': actor_id,
                'gender':actor_id%2 # 1:male 0:female
                })

        else:
            records.append({
                'features':feature_vector,
                'label':int(emotion_code)-1, # CrossEntropyLoss (pytorch) expects [0-N-1] labels
                'actor_id': actor_id
            })

    return records








        
