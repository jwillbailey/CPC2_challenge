import argparse
import random

import numpy as np
import pandas as pd
import torch

# def make_disjoint_train_set(
#     full_df: pd.DataFrame, test_df: pd.DataFrame
# ) -> pd.DataFrame:
#     """Make a disjoint train set for given test samples."""
#     # make sure that the train and test sets are disjoint
#     # i.e. no signals, systems or listeners are shared
#     train_df = full_df[~full_df.signal.isin(test_df.signal)]
#     train_df = train_df[~train_df.system.isin(test_df.system)]
#     train_df = train_df[~train_df.listener.isin(test_df.listener)]
#     assert not set(train_df.signal).intersection(set(test_df.signal))
#     return train_df



def get_disjoint_val_set(args, data):

    # validation has following values:
    # 0: training set
    # 1: disjoint validation set (listener) note: actually [1, 3, 5, 7]
    # 2: disjoint validation set (system) note: actually [2, 3, 6, 7]
    # 3: disjoint validation set (listener, system) note: actually [3, 7]
    # 4: disjoint validation set (scene) note: actually [4, 5, 6, 7]
    # 5: disjoint validation set (listener, scene) note: actually [5, 7]
    # 6: disjoint validation set (system, scene) note: actually [6, 7]
    # 7: disjoint validation set (listener, system, scene)

    data['validation'] = 0

    # Find listeners with the highest number of data points in subset CEC2
    val_listeners = data[data.subset == "CEC2"].listener.unique()
    val_listeners.sort()
    val_listeners = val_listeners[-2:]
    # print("val_listeners:", val_listeners)
    # Mark data associated with these listeners
    data.loc[data.listener.isin(val_listeners), 'validation'] += 1
    
    # Find the systems the validatin listeners used with the highest total number of data points
    val_systems = data[(data.validation == 1) & (data.subset == "CEC2")].system.value_counts(ascending = False).index[0:2]
    # print("val_systems:", val_systems)
    # Mark data associated with these systems
    data.loc[data.system.isin(val_systems), 'validation'] += 2

    # Find the scenes associated with the listener/system combinations
    val_scenes = data[(data.validation == 3) & (data.subset == "CEC2")].scene.value_counts(ascending = False).index
    # print("val_scenes:", val_scenes)
    data.loc[data.scene.isin(val_scenes), 'validation'] += 4
    
    vals = data.validation.value_counts()

    return data

# def extract_whisper_spec(data, args, theset):




def main(args):

    data = pd.read_json(args.in_json_file)
    data["subset"] = "CEC1"
    data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
    data2["subset"] = "CEC2"
    data = pd.concat([data, data2])
    data["predicted"] = np.nan  # Add column to store intel predictions

    data = get_disjoint_val_set(args, data)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_json_file", help="location of CEC1 metadata file", default = "/home/acp20rm/exp/data/clarity_CPC2_data/clarity_data/metadata/CEC1.train.1.json" # desktop
        # "--in_json_file", help="location of CEC1 metadata file", default = "~/data/clarity_CPC2_data/clarity_data/metadata/CEC1.train.1.json" # work laptop
    )
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)
    