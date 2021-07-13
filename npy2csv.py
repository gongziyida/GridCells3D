import glob
import numpy as np
import pandas as pd

PATH = 'data/*/*'

files = glob.glob(PATH)
for f in files:
    if f[-3:] == 'csv':
        continue
    data = np.load(f)
    pd.DataFrame(data).to_csv('data_csv' + f[4:-3] + 'csv', index=False)
