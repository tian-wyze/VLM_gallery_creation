import pandas as pd
import sys
import os

if __name__ == '__main__':

    # list all csv files in results/
    # folder = ''
    files = [f for f in os.listdir('results/') if f.endswith('.csv')]
    files.sort() # sort the files alphabetically

    for file in files:
        result = pd.read_csv(os.path.join('results', file))

        # calculate the acc based on the label and prediction columns
        acc = (result['label'] == result['prediction']).mean() * 100
        print(f'{file}: ')
        print(f'Acc: {round(acc, 1)}')
        print('-----------------------------')


