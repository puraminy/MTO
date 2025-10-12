import os
import glob
import pandas as pd
import argparse
from metrics.metrics import do_score
import time
from bleurt import score

def process_dataframe(df, spath, scorer):
    # Example process: drop columns with '_new' postfix
    scores = do_score(df, "bleu", spath, reval=True, scorer=scorer) 
    return df

def find_and_process_tsv_files(directory):
    # Use glob to find all .tsv files in the directory and subdirectories
    tsv_files = glob.glob(os.path.join(directory, '**', '*.tsv'), recursive=True)
    N = len(tsv_files)

    checkpoint = "/home/ahmad/pret/bleurt-20"
    bleu_scorer = score.BleurtScorer(checkpoint)
    for i, file in enumerate(tsv_files):
        # Read each .tsv file into a DataFrame
        df = pd.read_csv(file, sep='\t')

        # Process the DataFrame
        df = process_dataframe(df, file, bleu_scorer)

        # Save the processed DataFrame back to the file (or another location)
        # df.to_csv(file, sep='\t', index=False)
        print(f"{i}/ {N}) Processed and saved: {file}")
        time.sleep(20)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Find and process TSV files in a directory and its subdirectories.")
    parser.add_argument("path", help="Path to the directory to search for TSV files")
    
    # Parse arguments
    args = parser.parse_args()
    directory = args.path
    if directory == "adr":
        with open("/home/ahmad/temp/address.txt") as f:
            dirs = f.readlines()
    else:
        directory = os.path.join("/home/ahmad/logs", directory)
        dirs = [directory]

    # Find and process TSV files
    for directory in dirs:
        directory = directory.strip()
        print("Processing " + directory)
        find_and_process_tsv_files(directory)

if __name__ == "__main__":
    main()

