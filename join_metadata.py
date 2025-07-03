import pandas as pd

train=pd.read_parquet("Data\\train_data.parquet", engine="fastparquet")
offers=pd.read_parquet("Data\\offer_metadata.parquet", engine="fastparquet")

offers["id3"] = offers["id3"].astype(str)

# C1) Pick only the columns you care about from the offers table
offer_info = offers[[
    "id3",    # offer_id
    "f376",   # discount rate
    "f375",   # redemption frequency
    "id11",   # brand name
    "id10",   # industry code
]]

# C2) One‐hot encode brand and industry
offer_info = pd.get_dummies(
    offer_info,
    columns=["id11","id10"],
    prefix=["brand","industry"]        #Turns each unique id11 (brand) into a binary column, same for id10 (industry).
)

# C3) Merge these offer attributes into train
train = train.merge(offer_info, how="left", on="id3")

print(train.head())