from ad_causal_doubleml.config.paths import DATA_DIR
import pandas as pd

file_path = DATA_DIR / "train.log.txt"
from ad_causal_doubleml.feature_engineering.preprocessing import transform_hour

print('hello')



#2 min runtime

# explicit dtypes reduce memory usage when loading
# the large dataset
dtypes = {
    "click": "int8",
    "weekday": "int8",
    "hour": "int8",
    "logtype": "int8",

    "region": "int16",
    "city": "int16",

    "adexchange": "category",

    "slotwidth": "int16",
    "slotheight": "int16",

    "slotvisibility": "category",
    "slotformat": "int16",

    "slotprice": "int16",

    "bidprice": "int16",
    "payprice": "int16",

    "advertiser": "category",

    # High-cardinality categorical features
    "bidid": "string",
    "timestamp": "string",
    "ipinyouid": "string",
    "useragent": "category",
    "IP": "string",
    "domain": "string",
    "url": "string",
    "urlid": "string",
    "slotid": "string",
    "creative": "string",
    "keypage": "string",
    "usertag": "string"
}

print('loading df')
df = pd.read_csv(
    file_path,
    sep="\t",
    dtype=dtypes,
    na_values=["null"]
)
print('finished loading df')

# columns to drop 
columns_to_drop=['bidid','logtype','ipinyouid','IP','url','urlid','payprice']

# variables to one hot encode 
variables_one_hot = ['adexchange', 'useragent','weekday','region','slotwidth','slotheight','slotvisibility','slotformat','bidprice', 'advertiser','keypage']

# multi label columns to one hot encode 
variables_multi_label_one_hot_encode = 'usertag'

# columns to target encode 
target_categorical_features = ['city','domain','slotid','slotprice','creative']


df = transform_hour(df, 'hour')