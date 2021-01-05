#!/bin/sh
# Prepares UBIRISPr label files for cleaning in pandas.

cd /Users/maxwiese/Documents/DSI/assignments/The-Window/data #/path/to/data
mkdir Periocular_Labels
mv UBIRISPr/*.txt Periocular_Labels
cd Periocular_Labels
perl -i -pe 's/^.//' ./* # remove leading space from each row
perl -i -pe 's/Size;/Height;Width;Channel;/g' ./* # correct mappting for Size
perl -i -pe 's/;\n/,/g' ./* # transoform to single row with ',' and ';'
perl -i -pe 's/SizeY,/SizeY\n/g' ./* # split row at end of column names
perl -i -pe 's/;/,/g' ./* # repalce replace enduring ';' delim with ','
sed -i '' -- '$ s/.$//' * # remove final char, a hanging delim
