import pandas as pd

# 1) Load tables into pandas DataFrames
train = pd.read_parquet("Data\\train_data.parquet", engine="fastparquet")
events = pd.read_parquet("Data\\add_event.parquet", engine="fastparquet")

# print(train.info())
# print(event.info())

# Always do this before merging
train["id2"] = train["id2"].astype(str)
events["id2"] = events["id2"].astype(str)
train["id3"] = train["id3"].astype(str)
events["id3"] = events["id3"].astype(str)


# A1) Create two helper columns in events:
events["impression_flag"] = 1
#   • Every row in `events` is one impression, so we mark it with a 1.

events["click_flag"] = events["id7"].notna().astype(int)
#   • `id7` is the click timestamp; if it’s non-null, that event was a click.
#   • .notna() produces True/False; .astype(int) turns it into 1 (click) or 0 (no click).

# Convert to datetime (safe even if they're already datetime)
events["id4"] = pd.to_datetime(events["id4"], errors='coerce')
events["id7"] = pd.to_datetime(events["id7"], errors='coerce')

# A2) Group by customer & offer, then aggregate
evt_agg = (
    events
    .groupby(["id2","id3"])       # id2 = customer_id, id3 = offer_id
    .agg(
        total_impressions = ("impression_flag", "sum"),
        total_clicks      = ("click_flag",      "sum"),
        first_imp_time    = ("id4",             "min"),
        last_click_time   = ("id7",             "max"),
    )
    .reset_index()       # .reset_index(): turns the group keys id2, id3 back into regular columns.
)

# A3) Derive a click‐through rate feature
evt_agg["ctr"] = evt_agg["total_clicks"] / evt_agg["total_impressions"]

# A4) Merge these event‐level aggregates back into your main train table
train = train.merge(
    evt_agg,
    how="left",        #.merge(…, how="left") keeps every row from train.
    on=["id2","id3"]
)

# A5) Fill missing values (customers/offers with no events)
train[["total_impressions","total_clicks","ctr"]] = (                   #Wherever there was no matching row in evt_agg, pandas put NaN.
    train[["total_impressions","total_clicks","ctr"]]                   #We replace those NaNs with 0 (means “never saw” or “never clicked”).
    .fillna(0)
)

print(train.head())   