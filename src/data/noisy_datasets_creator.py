import pandas as pd

df = pd.read_pickle('../../data/NoisyAG-News.pkl')
print(len(df), 'rows')
print(df.columns.tolist())

print(df.head())
chosen_cols = [ 'label', 'gt',  'text' ]
df1 = df.rename(columns={'human_best_label': 'label', 'ground_truth': 'gt'})
print("DF1 COLUMNS: ",df1.columns.tolist())
df1[chosen_cols].to_csv(r'noise_best.csv', header=None, index=None, sep=' ', mode='a')


chosen_cols = [ 'label', 'gt',  'text' ]
df1 = df.rename(columns={'human_best_label': 'label', 'ground_truth': 'gt'})
print("DF1 COLUMNS: ",df1.columns.tolist())
df1[chosen_cols].to_csv(r'noise_best.csv', header=None, index=None, sep=' ', mode='a')


chosen_cols3 = ['label','gt', 'text']
df2 = df.rename(columns={'human_worst_label': 'label', 'ground_truth': 'gt'})
df2[chosen_cols3].to_csv(r'noise_worst.csv', header=None, index=None, sep=' ', mode='a')