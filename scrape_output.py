"""
Concatenates results from the two submodules for each individual target into a single csv in Files/
"""

import glob

import pandas as pd


# def _csv_reader(f):
#     row = pd.read_csv(f, header=None, squeeze=True, index_col=0)
#     return row


path = "Files/results/**/*findex.csv"
findex_files = glob.glob(path)
df = pd.read_csv(findex_files[0])

for i in range(1, len(findex_files)):
	   df_new = pd.read_csv(findex_files[i])
	   df = pd.concat([df, df_new])

df.to_csv("Files/findex.csv", index=False)

path = "Files/results/**/*globalpars.csv"
globalpars_files = glob.glob(path)
df = pd.read_csv(globalpars_files[0])

for i in range(1, len(globalpars_files)):
    df_new = pd.read_csv(globalpars_files[i])
    df = pd.concat([df, df_new])

df.to_csv("Files/globalpars.csv", index=False)
