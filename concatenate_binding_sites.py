import glob
import pandas as pd
import re
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--folder', type=str, default='./test_output', help='Location of the output folder of your p2rank run')
    parser.add_argument('-o', '--output', type=str, default='./binding_sites.csv', help='Output file of the concatenated binding sites')

    args = parser.parse_args()
    print(args)

    files = glob.glob(os.path.join(args.folder, '*.pdb_predictions.csv'))
    counter = 0
    uacs = []
    centers = []
    ranks = []
    for f in files:
        try:
            fullname = f.split('/')[-1]
            identifier = fullname.split('-')[1]
            df = pd.read_csv(f)
            center_x = df['   center_x'].to_numpy()
            center_y = df['   center_y'].to_numpy()
            center_z = df['   center_z'].to_numpy()
            if center_x.size:
                for enum, i, in enumerate(zip(center_x, center_y, center_z)):
                    rank = f'{identifier}_{enum+1}'
                    uacs.append(identifier)
                    centers.append(i)
                    ranks.append(enum)

        except IndexError:
            print(f'could not process {f}')
            pass

    counter += 1
    if counter % 1000 == 0:
        print(f'processed {counter} pockets')

    pd.DataFrame(zip(uacs, centers, ranks), columns=['identifier', 'binding_site', 'rank']).to_csv(args.output, index=False)

