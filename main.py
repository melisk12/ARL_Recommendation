
#!pip install mlxtend
import pandas as pd
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


# --------------- Data Preparing ---------------

data = pd.read_csv("case_armut/armut_data.csv")
df = data.copy()
df.head()
df.info()
df.shape
df.describe().T

# ServiceID represents a different service for each CategoryID. We create a new variable to represent services by combining ServiceID and CategoryID with "_".

df["hizmetler"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head()


# Adım 3: The data set consists of the date and time the services were received, there is no basket definition (invoice, etc.).
# We are asked to create this. For this;
# In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Here, the basket definition is the services that each customer receives monthly.
# For example; The customer with ID 7256 received a basket of 9_4, 46_4 services in the 8th month of 2017;
# The 9_4 and 38_4 services received in the 10th month of 2017 represent another basket.
# Baskets must be identified with a unique ID.
# To do this, first we will create a new date variable that contains only the year and month.
# We will combine the User ID and the date variable you just created with "_" and assign it to a new variable called ID.

df.dtypes
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.dtypes

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m") 
df["sepet_id"]= df["UserId"].astype(str)+"_" + df["New_Date"].astype(str)
df.head()


# --------------- Create Association Rules ---------------


# Let's create the basket service pivot table

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


df.groupby(['sepet_id', 'hizmetler'])['hizmetler'].count().unstack().fillna(0).head()

# OR:
invoice_product_df =df.groupby(['sepet_id', 'hizmetler'])['hizmetler'].count().unstack().fillna(0).applymap(lambda x:1 if x>0 else 0)
invoice_product_df.head()

# The unstack() function converts a multi-level index into a single-level index. 
# When using the unstack() function, we should be careful that missing data may occur as a result of this operation. 
# The stack() function reverses this process, that is, it turns a flattened DataFrame back into a multi-level index.
# In fact, by writing 0 and 1, we are performing binary encode.

# Let's create association rules.

frequent_itemsets = apriori(invoice_product_df,
                            min_support=0.01,
                            use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]
rules.head()


# arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 3)


