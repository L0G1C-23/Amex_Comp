import pandas as pd

# 1) Load your tables into pandas DataFrames
train  = pd.read_parquet("Data\\train_data.parquet", engine="pyarrow")
trans  = pd.read_parquet("Data\\add_trans.parquet", engine="pyarrow")

trans["id2"] = trans["id2"].astype(str)

# B1) Convert the transaction‐date column to pandas datetime
trans["txn_date"] = pd.to_datetime(trans["f370"])

# B2) Aggregate per customer (id2)
cust_txn = (
    trans
    .groupby("id2")
    .agg(
        avg_txn_amt   = ("f367", "mean"),            # average transaction amount
        last_txn_date = ("txn_date", "max"),         # most recent transaction date
        txn_count     = ("f367", "count"),           # total number of transactions
    )
    .reset_index()
)

# B3) Merge transaction features into train
train = train.merge(cust_txn, how="left", on="id2")

# Fill missing with 0 (customers with no past transactions)
train[["avg_txn_amt","txn_count"]] = train[["avg_txn_amt","txn_count"]].fillna(0)

print(train.head())