import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filename = "nl_energy_data_2025.csv"

df = pd.read_csv(filename, index_col=0, parse_dates=True)

plt.figure(figsize=(12, 6))

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.title('Missing Data Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('missing_data_heatmap.png')
    