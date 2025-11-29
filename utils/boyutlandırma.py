from PIL import Image
import os

# Giriş ve çıkış klasörlerini belirle
giris_klasoru = "C:/Users/STUDYSPC/Desktop/veri_setim/yeniveri"
cikis_klasoru = "C:/Users/STUDYSPC/Desktop/maske_tespit/yeniverii"

os.makedirs(cikis_klasoru, exist_ok=True)

for dosya in os.listdir(giris_klasoru):
    if dosya.endswith(".jpg") or dosya.endswith(".png"):
        img_yolu = os.path.join(giris_klasoru, dosya)
        img = Image.open(img_yolu)

        # Orijinal boyutları al
        genislik, yukseklik = img.size

        # En uzun kenara göre yeni kare canvas oluştur
        uzun_kenar = max(genislik, yukseklik)
        yeni_img = Image.new("RGB", (uzun_kenar, uzun_kenar), (0, 0, 0))  # Siyah arka plan

        # Görseli ortalayarak kare canvas’a yapıştır
        sol = (uzun_kenar - genislik) // 2
        ust = (uzun_kenar - yukseklik) // 2
        yeni_img.paste(img, (sol, ust))

        # Son olarak yeniden boyutlandır (örneğin 512x512)
        kare_512 = yeni_img.resize((512, 512))

        # Yeni görseli kaydet
        kayit_yolu = os.path.join(cikis_klasoru, dosya)
        kare_512.save(kayit_yolu)

print("Tüm görseller 512x512 kare formatta dönüştürüldü.")
