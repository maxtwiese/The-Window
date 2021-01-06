"""Intakes label files and combines into single .csv for UBIRISPr.

01-05-2021: Need to test RoI boxes for eye detection and will then
            refine output .csv(s).

01-06-2021: Formatting of the label files was giving pandas (and me,
            of course) a headache, so I made a selective reader.
"""
from glob import glob
import numpy as np
import os
import pandas as pd

def liu_ROI(x1, y1, x2, y2, a=0.5, b=0.5):
    """Boundary Box (BB) algorithm.

    Algorithm for BB from canthus point location based on work of Liu,
    et al. (2017). Finds midpoint of canthus points then constructs BB
    centered at midpoint with size as a proportion of Euclidean
    Distance.

    Parameters
    ----------
    x1, y1, x2, y2 : Integer
        Coordinates of inner and outer canthus points.
    a : Float
        Default = 0.5. Multiplier that works in conjuncition with
        Euclidean Distance to decide width of BB
    b : Float
        Default = 0.5. Multiplier that works in conjuncition with
        Euclidean Distance to decide height of BB
    Returns
    -------
    x3, y3, x4, y4
    """
    euc_dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    x3, y3 = mid_x - a * euc_dis, mid_y - b * euc_dis
    x4, y4 = mid_x + a * euc_dis, mid_y + b * euc_dis
    return x3, y3, x4, y4

if __name__ == '__main__':
    os.chdir('/Users/maxwiese/Documents/DSI/assignments/The-Window/data'\
             '/UBIRISPr')
    head = ['FileName','CornerOutPtX', 'CornerOutPtY', 'CornerInPtX',
            'CornerInPtY']
    txts = sorted(glob('*.txt'))
    file_name= []
    out_pt_x = []
    out_pt_y = []
    in_pt_x = []
    in_pt_y = []

    for txt in txts: # Extract canthus data from given labels files.
        file_name.append(txt) # Used to drop duplicates.
        fp = open(txt)
        for i, line in enumerate(fp):
            if i == 14:
                out_pt_x.append(line)
            elif i == 15:
                out_pt_y.append(line)
            elif i == 18:
                in_pt_x.append(line)
            elif i == 19:
                in_pt_y.append(line)
            elif i > 19:
                break
        fp.close()

    # Transform extracted data as DataFrame.
    zip_list = list(zip(file_name, out_pt_x, out_pt_y, in_pt_x, in_pt_y))
    df = pd.DataFrame(zip_list, columns=head).drop_duplicates()
    df['FileName'] = sorted(glob('*.jpg')) # Replace with image files names.
    #df.insert(0, 'ImageName', sorted(glob('*.jpg'))) # Insert image file names.

    # Strip coordinates and convert to integers.
    head.remove('FileName')
    for col in head: 
        df[col] = df[col].str.strip(' ;\n').astype('int')

    # Construct boundary boxes and save coordinates as new columns.
    df['LeftBB'], df['LowerBB'], df['RightBB'], df['UpperBB'] = \
        liu_ROI(df['CornerOutPtX'], df['CornerOutPtY'], df['CornerInPtX'],
                df['CornerInPtY'], a=.5, b=.5)

    df.to_csv(r'../UBIRISPr_Labels.csv', index=False) # Output to .csv
