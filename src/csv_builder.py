"""Intakes label files and combines into single .csv for UBIRISPr.

If you are getting an error: [Errno 13] Permission denied:
"./labelwrangler.sh" then run: $ chmod u+rx ./labelwrangler.sh

01-05-2021: Need to test RoI boxes for eye detection and will then
            refine output .csv(s).
"""
import numpy as np
import os
import pandas as pd
import subprocess

subprocess.run("chmod u+rx ./labelwrangler.sh", shell=True) # Permissions.
subprocess.call("./labelwrangler.sh") # Call BASH preprocessing script.

path_to_data = '/Users/maxwiese/Documents/DSI/assignments/The-Window/data/'
dirpath = path_to_data + 'Periocular_Labels'
output = path_to_data + 'UBIRISPr_labels.csv'
combined_cols = ['PoseAngle', 'GazeAngle', 'Pigm', 'EyeClosure',
    'HairOcclusion', 'Glass', 'CornerOutPtX', 'CornerOutPtY','IrisCenterPtX',
    'IrisCenterPtY', 'CornerInPtX', 'CornerInPtY','EyebrowOutPtX',
    'EyebrowOutPtY', 'EyebrowMidPtX', 'EyebrowMidPtY','EyebrowInPtX',
    'EyebrowInPtY', 'EyeSizeX', 'EyeSizeY'] # Left/Right agnostic features.

df = pd.concat(pd.read_csv(os.path.join(dirpath, fname)) \
    for fname in sorted(os.listdir(dirpath))
    ).reset_index(drop=True) # Read labels into a DataFrame.

for col in combined_cols: # Create columns for agnostic features and drop.
    df[col] = np.where(df['R' + col].notna(), df['R' + col], df['L' + col])
    del df['R' + col], df['L' + col]

df.drop(columns=['CurrentDir', 'FileName', 'Unnamed: 27'], inplace=True)
df.to_csv(output, index=False) # Output to .csv
