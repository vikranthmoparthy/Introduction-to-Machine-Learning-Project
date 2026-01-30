"""
Most of this code was adapted from Week 2's Practical: Data Preprocessing
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


filename = "nl_energy_data_2025.csv"

#Load the CSV data into a dataframe and convert times into date-time objects
df = pd.read_csv(filename, index_col=0, parse_dates=True)

plt.figure(figsize=(12, 6)) #Create the figure dimensions

#Generating the heatmap using the practical code
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.title('Missing Data Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right') #Rotating x-axis labels so they don't overlap
plt.tight_layout() #Automatically adjusting margins

plt.savefig('missing_data_heatmap.png')
    