#!/bin/sh
# Builds a .csv for ETL of images and labels from UBIRIS Periodcular
# Dataset

# Move label files (.txt) to their own directory.
cd /Users/maxwiese/Documents/DSI/assignments/The-Window/data #/path/to/data
mkdir Periocular_Labels
mv UBIRISPrSmall/*.txt Periocular_Labels

# Preprocess files for pandas.
cd Periocular_Labels
perl -i -pe 's/^.//' ./*
perl -i -pe 's/Size;/Size1;Size2;Size3;/g' ./*
perl -i -pe 's/;\n/,/g' ./*
perl -i -pe 's/SizeY,C/SizeY\nC/g' ./*
perl -i -pe 's/;/,/g' ./*
sed -i '' -- '$ s/.$//' *

# Extract labels from label files.
function csv_builder {
python -  <<END
import os
import pandas as pd

data = '/Users/maxwiese/Documents/DSI/assignments/The-Window/data/'
dirpath = data + 'Periocular_Labels'
output = data + 'labels.csv'

pd.concat(
pd.read_csv(os.path.join(dirpath, fname))
for fname in sorted(os.listdir(dirpath))
).to_csv(output)
END
}

# Run like the wind.
csv_builder
