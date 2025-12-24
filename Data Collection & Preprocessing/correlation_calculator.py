import pandas as pd

def main():
    filename = 'nl_energy_data_2025.csv'
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    corr_matrix = df.corr()
    print(corr_matrix.round(2))

if __name__ == "__main__":
    main()