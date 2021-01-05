#!/bin/sh
# Builds a .csv for labels of UBIRIS Periodcular Dataset.

# Move label files (.txt) to their own directory.
cd /Users/maxwiese/Documents/DSI/assignments/The-Window/data #/path/to/data
mkdir Periocular_Labels
mv UBIRISPr/*.txt Periocular_Labels

# Preprocess files for pandas.
cd Periocular_Labels
perl -i -pe 's/^.//' ./* # remove leading space from each row
perl -i -pe 's/Size;/Size1;Size2;Size3;/g' ./* # correct mappting for Size
perl -i -pe 's/;\n/,/g' ./* # transoform to single row with ',' and ';'
perl -i -pe 's/SizeY,C/SizeY\nC/g' ./* # split row at end of column names
perl -i -pe 's/;/,/g' ./* # repalce replace enduring ';' delim with ','
sed -i '' -- '$ s/.$//' * # remove final char, a hanging delim

# Extract labels from label files to .csv
function csv_builder {
python -  <<END
import os
import pandas as pd

data = '/Users/maxwiese/Documents/DSI/assignments/The-Window/data/'
dirpath = data + 'Periocular_Labels'
output = data + 'UBIRISPr_labels.csv'

pd.concat(
pd.read_csv(os.path.join(dirpath, fname))
for fname in sorted(os.listdir(dirpath))
).reset_index().drop(columns=['index']).to_csv(output)
END
}

# Run like the wind.
csv_builder
