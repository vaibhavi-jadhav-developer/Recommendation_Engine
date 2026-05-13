import pandas as pd
import random

df = pd.read_csv('data/cleaned_data.csv')

test_df = df.groupby('user_id').tail(1).copy()
train_df = df.drop(test_df.index)

print('Generating negative samples. This might take 10-20 seconds')

user_interacted_items = df.groupby('user_id')['item_id'].apply(set).to_dict()
num_items = df['item_id'].nunique()

train_data = []
num_negatives = 4 

for row in train_df.itertuples():
    user = row.user_id
    item = row.item_id
    train_data.append([user, item, 1])
    
    for _ in range(num_negatives):
        while True:
            neg_item = random.randint(0, num_items - 1)
            if neg_item not in user_interacted_items[user]:
                break
        train_data.append([user, neg_item, 0])


final_train_df = pd.DataFrame(train_data, columns=['user_id', 'item_id', 'interaction'])

final_train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

print(f'Training Data size: {len(final_train_df)} rows')
print(f'Test Data size: {len(test_df)} rows')
print("Success! 'train_data.csv' and 'test_data.csv' are ready for Deep Learning.")