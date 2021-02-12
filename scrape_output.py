"""
Concatenates results from the two submodules for each individual target into a single csv in Files/.
"""

import glob

import pandas as pd


def main():
    """Concatenates results from the two submodules for each
    individual target into a single csv in Files/.
    """
    # Concatenating findex files i.e. find excess results
    path = "Files/results/**/*findex.csv"
    findex_files = glob.glob(path)
    if findex_files:
        findex_df = pd.read_csv(findex_files[0])
        for i in range(1, len(findex_files)):
            df_new = pd.read_csv(findex_files[i])
            findex_df = pd.concat([findex_df, df_new])

        findex_df.to_csv("Files/findex.csv", index=False)

    # Concatenating globalpars files i.e. fit background results
    path = "Files/results/**/*globalpars.csv"
    globalpars_files = glob.glob(path)
    if globalpars_files:
        globalpars_df = pd.read_csv(globalpars_files[0])
        for i in range(1, len(globalpars_files)):
            df_new = pd.read_csv(globalpars_files[i])
            globalpars_df = pd.concat([globalpars_df, df_new])

        globalpars_df.to_csv("Files/globalpars.csv", index=False)


if __name__ == "__main__":
    main()
