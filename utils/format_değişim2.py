import os

ana_klasor = "C:/Users/STUDYSPC/Desktop/veri_setim/yeniveri"  # Tüm klasörlerin içinde olduğu ana klasörü buraya yaz

for klasor_adi, alt_klasorler, dosyalar in os.walk(ana_klasor):
    for dosya in dosyalar:
        if dosya.lower().endswith((".jpeg", ".png", ".jpg")):
            eski_yol = os.path.join(klasor_adi, dosya)
            dosya_adi, _ = os.path.splitext(dosya)
            yeni_dosya_adi = dosya_adi + ".jpg"
            yeni_yol = os.path.join(klasor_adi, yeni_dosya_adi)

            os.rename(eski_yol, yeni_yol)

print("Tüm görsellerin uzantısı .jpg olarak değiştirildi.")
