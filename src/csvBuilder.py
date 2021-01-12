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

if __name__ == '__main__':
    #os.chdir('/Users/maxwiese/Documents/DSI/assignments/The-Window/data'\
    #         '/UBIRISPr')
    os.chdir('../data/UBIRISPr')
    head = ['FileName', 'CornerOutPtX', 'CornerOutPtY', 'CornerInPtX',
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
    df['Y1'] = df['Y1'] /  df['Height'] * 256
    df['Y2'] = df['Y2'] / df['Height'] * 256
    df['X1'] = (2 * df['X1'] + df['Width'] - df['Height']) / df['Height'] * 128
    df['X2'] = (2 * df['X2'] + df['Width'] - df['Height']) / df['Height'] * 128
    
    df_small = df.head(1000)
#    df.to_csv('~/The-Window/data/UBIRISPr_Labels.csv', index=False) # ubuntu
#    df_small.to_csv('~/The-Window/data/UBIRISPr_Labels_small.csv', index=False)
    
    #print("""
############################################
#           Data Set Information           #
############################################\n""")
    #print(df.info())
    #print("--------------------------------------------\n")
    #print("""
############################################
#        Data Set Small Information        #
############################################
#Note: This is the first 100 entries for
#complexity reduction in testing.\n""")
#    print(df_small.info())
    
    #msk = np.random.rand(len(df)) < 0.1
    #train = df[msk]
    #test = df[~msk]
    
    # Make it small for now. As above, so below.
    msk1 = np.random.rand(len(df)) < 0.8
    train = df[msk1]
    test = df[~msk1]
    msk2 = np.random.rand(len(df_small)) < 0.8
    train_small = df_small[msk2]
    test_small = df_small[~msk2] 

    #print(f"Train Set Informaiton:\n {train.info()}")
    #print(f"Test Set Informaiton:\n {test.info()}")

    ############################################
    #          Train Set Information.          
    ############################################
    # Int64Index: 1649 entries, 4 to 10198
    # Data columns (total 11 columns):
    #  #   Column        Non-Null Count  Dtype  
    # ---  ------        --------------  -----  
    #  0   FileName      1649 non-null   object 
    #  1   CornerOutPtX  1649 non-null   int64  
    #  2   CornerOutPtY  1649 non-null   int64  
    #  3   CornerInPtX   1649 non-null   int64  
    #  4   CornerInPtY   1649 non-null   int64  
    #  5   X1            1649 non-null   float64
    #  6   Y1            1649 non-null   float64
    #  7   X2            1649 non-null   float64
    #  8   Y2            1649 non-null   float64
    #  9   Width         1649 non-null   int64  
    #  10  Height        1649 non-null   int64  
    # dtypes: float64(4), int64(6), object(1)
    # memory usage: 154.6+ KB
    ############################################
    
    ############################################
    #          Test Set Information.          
    ############################################
    #  Int64Index: 398 entries, 52 to 10109
    #  Data columns (total 11 columns):
    #  #   Column        Non-Null Count  Dtype  
    # ---  ------        --------------  -----  
    #  0   FileName      398 non-null    object 
    #  1   CornerOutPtX  398 non-null    int64  
    #  2   CornerOutPtY  398 non-null    int64  
    #  3   CornerInPtX   398 non-null    int64  
    #  4   CornerInPtY   398 non-null    int64  
    #  5   X1            398 non-null    float64
    #  6   Y1            398 non-null    float64
    #  7   X2            398 non-null    float64
    #  8   Y2            398 non-null    float64
    #  9   Width         398 non-null    int64  
    #  10  Height        398 non-null    int64  
    # dtypes: float64(4), int64(6), object(1)
    # memory usage: 37.3+ KB
    ############################################

    train.to_csv('~/The-Window/data/Train_Set.csv', index=False) # ubuntu
    test.to_csv('~/The-Window/data/Test_Set.csv', index=False)
    train_small.to_csv('~/The-Window/data/Train_Set_small.csv', index=False) # ubuntu
    test_small.to_csv('~/The-Window/data/Test_Set_small.csv', index=False)
    