
#!pip install mlxtend
import pandas as pd
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.

data = pd.read_csv("case_armut/armut_data.csv")
df = data.copy()
df.head()
df.info()
df.shape
df.describe().T

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["hizmetler"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head()


# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Bunu oluşturmamız isteniyor. Bunun için;
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.

df.dtypes
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.dtypes

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m") # yıl ve ay içeren yeni bir date değişkeni
df["sepet_id"]= df["UserId"].astype(str)+"_" + df["New_Date"].astype(str)
df.head()


#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz.
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

# hizmetler sutunlarda sepetıd de satırlarda olmalıdır burda
df.groupby(['sepet_id', 'hizmetler'])['hizmetler'].count().unstack().fillna(0).head()
# 2.yol:
invoice_product_df =df.groupby(['sepet_id', 'hizmetler'])['hizmetler'].count().unstack().fillna(0).applymap(lambda x:1 if x>0 else 0)
invoice_product_df.head()
# unstack() fonksiyonu çok seviyeli indeksi tek seviyeli bir indekse dönüştürür
# unstack() işlevini kullanırken, bu işlemin sonucunda eksik veri oluşabileceğine dikkat etmelisiniz. stack() işlevi ise
# bu işlemi tersine çevirir, yani düzleştirilmiş bir DataFrame'i tekrar çok seviyeli bir indekse dönüştürür.
# diğer yandan 0 ve 1 yazarakta binary encode işlemi yapmış oluyoruz.

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(invoice_product_df,
                            min_support=0.01,
                            use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]
rules.head()


#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 3)


