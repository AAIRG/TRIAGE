import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from settings import ROOT_DIR

TABLES_DATA_DIR = os.path.join(ROOT_DIR, "data/post_processed/tables_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "figures")

# To embed fonts in PDF
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# --- File Paths ---
file = os.path.join(TABLES_DATA_DIR, "technique_value_counts.csv")
attack_techniques_file = os.path.join(ROOT_DIR, "data/pre_processed/attack_techniques_base.csv")
output_file = os.path.join(OUTPUT_DIR, "rank_frequency.csv")

# --- Data Processing ---
# Read the CSV file
df = pd.read_csv(file)

# don't filter out techniques with a count value of 2 or less
#df = df[df['count'] >= 2]

# Read the attack techniques base file
attack_df = pd.read_csv(attack_techniques_file)

# Filter attack techniques
attack_df = attack_df[(attack_df['category'] == 'technique') & (attack_df['attack_matrix'] == 'enterprise-attack')]

# Merge the two dataframes
merged_df = pd.merge(df, attack_df[['attack_id', 'attack_name']], on='attack_id', how='left')

# Select and save the desired columns
top_techniques_df = merged_df[['attack_id', 'count', 'attack_name']]
top_techniques_df.to_csv(output_file, index=False)


# --- Plotting ---
font_size = 10 * 1.4
plt.style.use("petroff10")
plt.rcParams.update({'font.size': font_size})
plt.figure(figsize=(10, 6))

# Sort by count
x = np.arange(1, len(df) + 1)
y = df['count'].to_numpy()
plt.vlines(x, 0, y, linewidth=1)   # stems only
plt.scatter(x, y, s=20, zorder=3)  # candies


# --- Customize and show the plot ---
plt.xlabel("rank")
plt.ylabel("frequency")
#plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rank-frequency.png"))
plt.savefig(os.path.join(OUTPUT_DIR, "rank-frequency.pdf"))
plt.show()