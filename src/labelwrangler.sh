#!/bin/sh
# Prepares UBIRISPr label files for cleaning in pandas.
# 01-06-2021: Written out for now. Keeping here behind glass I can
# break in case of fire.

cd /Users/maxwiese/Documents/DSI/assignments/The-Window/data #/path/to/data
mkdir Periocular_Labels
mv UBIRISPr/*.txt Periocular_Labels
cd Periocular_Labels
sed -i '' -- '1,6d' * # remove first 5 rows
perl -i -pe 's/^.//' ./* # remove leading space from each row
perl -i -pe 's/;\n/,/g' ./* # transoform to single row with ',' and ';'
sed -i '' -- '$ s/.$//' * # remove final char, a hanging delim
