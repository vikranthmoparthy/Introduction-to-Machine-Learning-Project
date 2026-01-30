"""
In this file, we use Pandas functionality to calculate the correlation between each pair of features in our data.
The results can be visualised in a heatmap, which is included in this folder.
"""
import pandas as pd

def main():
    filename = 'nl_energy_data_2025.csv'
    df = pd.read_csv(filename, index_col=0, parse_dates=True) #Load data and read CSV
    corr_matrix = df.corr() #We calculate the Pearson correlation coefficient between every pair of columns
    print(corr_matrix.round(2))

if __name__ == "__main__":
    main()