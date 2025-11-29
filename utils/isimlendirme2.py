import os

klasor_yolu = "C:/Users/STUDYSPC/Desktop/maske_tespit/yeniverii"  # 🔁 Klasör yolunu buraya yaz
baslangic_numarasi = 38  # ✅ Kaldığın numarayı buraya yaz (örn. çantadaa048'den sonra 49)

sayac = baslangic_numarasi

for dosya in sorted(os.listdir(klasor_yolu)):
    if dosya.endswith(".jpg") or dosya.endswith(".png"):
        eski_yol = os.path.join(klasor_yolu, dosya)
        yeni_ad = f"boyundaa{sayac:03}.jpg"  # 003 formatında olacak
        yeni_yol = os.path.join(klasor_yolu, yeni_ad)
        os.rename(eski_yol, yeni_yol)
        sayac += 1

print("Yeniden isimlendirme tamamlandı!")
