import category_encoders as ce
import pandas as pd

filename = "fraudTest"

df = pd.read_csv(filename + ".csv")

df2 = df[['cc_num', 'category', 'amt', 'gender', 'zip', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']].copy()

encoder = ce.OneHotEncoder(cols='category', use_cat_names=True, return_df=True)
df3 = encoder.fit_transform(df2)

encoder = ce.OneHotEncoder(cols='gender', return_df=True, use_cat_names=True)
df4 = encoder.fit_transform(df3)

df4.to_csv(filename + "Encoded.csv")
