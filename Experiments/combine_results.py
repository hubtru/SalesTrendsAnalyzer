import os
import sys

import pandas as pd


OUTPUT_PATH = "./outputs"
RESULT_SUFFIX = "results.csv"


def main():
    all_results = []
    for filename in os.listdir(OUTPUT_PATH):
        if filename.endswith(RESULT_SUFFIX):
            all_results.append(pd.read_csv(OUTPUT_PATH + "/" + filename, header=[0, 1]))
    if len(all_results) == 0:
        print("No result-files found")
        sys.exit(1)

    combined_results = pd.concat(all_results, ignore_index=True)

    # move experiment name to forst column:

    first_column = combined_results.pop(("Experiment", "Name"))
    combined_results.insert(1, ("Experiment", "Name"), first_column)

    combined_results.to_csv("all_results.csv")


if __name__ == "__main__":
    main()
