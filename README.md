# ARL_Recommendation

## İş Problemi
Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca ulaşılmasını sağlamaktadır.
Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


## VERİ SETİ
Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat bilgisini içermektedir.

#### UserId: Müşteri numarası
#### ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi) Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder. (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken
#### CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
#### CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
#### CreateDate: Hizmetin satın alındığı tarih



## PROJE GÖREVLERİ
### Görev 1: Veriyi Hazırlama
#### Adım 1: armut_data.csv dosyasını okutunuz.
#### Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.
#### Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
#### Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz
#### Adım 2: Birliktelik kurallarını oluşturunuz.
#### Adım3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

