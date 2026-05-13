import pandas as pd

columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('data/u.data', sep='\t', names=columns)


df['interaction'] = 1

df['user_id'] = df['user_id'].astype('category').cat.codes
df['item_id'] = df['item_id'].astype('category').cat.codes

df = df.sort_values(by=['user_id', 'timestamp'])

df = df[['user_id', 'item_id', 'interaction', 'timestamp']]

df.to_csv('data/cleaned_data.csv', index=False)

print(f"Total Unique Users: {df['user_id'].nunique()}")
print(f"Total Unique Movies: {df['item_id'].nunique()}")
print("Success! Cleaned data saved to 'data/cleaned_data.csv'")
print("\nFirst 5 rows:")
print(df.head())