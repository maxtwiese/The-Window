"""Intakes label files and combines into .csv(s) for UBIRISPr.

01-05-2021: Need to test RoI boxes for eye detection and will then
            refine output .csv(s).

01-06-2021: Formatting of the label files was giving pandas (and me,
            of course) a headache, so I made a selective reader.

01-12-2021: Continually adjusting the output over the last few days has
            still not bourn a certain enough need for outputs. As such,
            I am leaving in coded out chunks for my continued memory
            tuning. Will wrap into functions when narrowed.
"""
from glob import glob
import numpy as np
import os
import pandas as pd
from PIL import Image

def liu_RoI(x1, y1, x2, y2, a=0.5, b=0.5):
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

def full_DataFrame():
    """Constructs DataFrame of image paths and cooresponsing labels.
    
    Reads specific lines from formatted .txt label files to accompany
    images in the UBIRIS Periocular Dataset into an array. Builds a
    Dataframe from the array, appends image file names, and builds
    bounding box coordinates with liu_RoI function above. Constant
    multipliers are used for later rescaling of images to squares. All
    pictures are landscape so non scaling is written for landscape
    pictures only.
    
    Parameters
    ----------
    None
    Returns
    -------
    df : DataFrame
        10,199 Rows x 11 Columns
    """
    os.chdir('../data/UBIRISPr') 
    head = ['FileName', 'CornerOutPtX', 'CornerOutPtY', 'CornerInPtX',
            'CornerInPtY']
    txts = sorted(glob('*.txt')) # All Label files
    file_name= []
    out_pt_x = []
    out_pt_y = []
    in_pt_x = []
    in_pt_y = []

    for txt in txts: # Extract canthus data from given label files.
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

    # Strip coordinates and convert to integers.
    head.remove('FileName')
    for col in head:
        df[col] = df[col].str.strip(' ;\n').astype('int')

    # Construct boundary boxes and save coordinates as new columns.
    df['X1'], df['Y1'], df['X2'], df['Y2'] = liu_RoI(df['CornerOutPtX'],
                                                     df['CornerOutPtY'],
                                                     df['CornerInPtX'],
                                                     df['CornerInPtY'],
                                                     a=.5, b=.5)
    df['Width'] = df['FileName'].apply(lambda x: Image.open(x).size[0])
    df['Height'] = df['FileName'].apply(lambda x: Image.open(x).size[1])
    df['Y1'] = df['Y1'] /  df['Height'] * 256 # Aforementioned constant mult.
    df['Y2'] = df['Y2'] / df['Height'] * 256
    df['X1'] = (2 * df['X1'] + df['Width'] - df['Height']) \
        / df['Height'] * 128
    df['X2'] = (2 * df['X2'] + df['Width'] - df['Height']) \
        / df['Height'] * 128

    return df

if __name__ == '__main__':
    small = 1000 # Size of df to work with.
    df = full_DataFrame()
    df_small = df.head(small)

    # Make it small for now. As above, so below.
    msk1 = np.random.rand(len(df)) < 0.8
    train = df[msk1]
    test = df[~msk1]
    msk2 = np.random.rand(len(df_small)) < 0.8
    train_small = df_small[msk2]
    test_small = df_small[~msk2] 

    # For splitting here.
    train.to_csv('../Train_Set.csv', index=False) # ubuntu
    test.to_csv('../Test_Set.csv', index=False)
    train_small.to_csv('../Train_Set_small.csv', index=False) # ubuntu
    test_small.to_csv('../Test_Set_small.csv', index=False)
    
    # For splitting in DataLoader.
    #df.to_csv('../data/UBIRISPr_Labels.csv', index=False) # ubuntu
    #df_small.to_csv('../data/UBIRISPr_Labels_small.csv', index=False)
    
    # Visual sanity check.
    print("############################################\n" +
          "#           Data Set Information           #\n" +
          "############################################\n")
    print(df.info())
    print(f"\nTrain: {train.shape[0]} // Test: {test.shape[0]}")
    print("--------------------------------------------\n")
    print("############################################\n" +
          "#        Data Set Small Information        #\n" +
          "############################################\n" +
          "Note: This is the first {small} entries for\n" +
          "complexity reduction in testing.\n")
    print(df_small.info())
    print(f"\nTrain: {train_small.shape[0]} // Test: {test_small.shape[0]}")
    print("--------------------------------------------")
