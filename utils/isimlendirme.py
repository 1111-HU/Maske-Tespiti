import os

ana_klasor = "C:/Users/STUDYSPC/Desktop/maske_tespit"

for klasor_adi in os.listdir(ana_klasor):
    klasor_yolu = os.path.join(ana_klasor, klasor_adi)

    if os.path.isdir(klasor_yolu):
        sayac = 1
        for dosya in sorted(os.listdir(klasor_yolu)):
            if dosya.endswith(".jpg") or dosya.endswith(".png"):
                dosya_yolu = os.path.join(klasor_yolu, dosya)

                # Yeni dosya adını oluştur
                yeni_ad = f"{klasor_adi}{sayac:03}.jpg"
                yeni_yol = os.path.join(klasor_yolu, yeni_ad)

                # Dosyayı yeniden adlandır
                os.rename(dosya_yolu, yeni_yol)
                sayac += 1

print("Tüm görseller yeni isimlere göre yeniden adlandırıldı.")
