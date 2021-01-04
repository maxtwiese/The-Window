#!/bin/sh
# Builds a .csv for ETL of images and labels from UBIRIS Periodcular
# Dataset. Run in the directory containing 'UBIPeriocular'.

# Move label files (.txt) to their own directory.
mkdir Periocular_Labels
mv UBIPeriocular/*.txt Periocular_Labels

# Extract labels from label files.
function csv_builder {
python -  <<END
import os
import pandas as pd

data = '/Users/maxwiese/Documents/DSI/assignments/The-Window/data/'
dirpath = data + 'Periocular_Labels'
output = data + 'labels.csv'

pd.concat(
pd.read_csv(os.path.join(dirpath, fname), sep=';')
for fname in sorted(os.listdir(dirpath))
).to_csv(output)
END
}

# Run like the wind.
csv_builder
