# Deteksi Masker dengan YOLOv8

Deteksi masker wajah secara real-time (pakai masker / tidak) menggunakan YOLOv8, dilatih di dataset publik Roboflow yang kecil, lalu diperbaiki lewat lima iterasi model setelah pengujian di foto CCTV nyata menunjukkan celah yang tidak terlihat dari angka validasi saja.

Repo ini mendokumentasikan seluruh proses iterasinya, termasuk versi yang justru memperburuk keadaan, karena itu catatan yang lebih jujur dan lebih berguna dibanding cuma menampilkan hasil akhir.

## Isi

- `Roboflow_Model_Yolo_Mask.ipynb` — download dataset, rebalancing kelas, dan training untuk kelima versi model (v1–v5)
- `Test_Model_Yolo_Mask.ipynb` — inferensi, eksperimen threshold/resolusi, dan demo deployment via MQTT
- `assets/` — contoh hasil deteksi pada foto jalanan nyata (lihat di bawah)

## Masalah & Dataset

Dataset dasarnya adalah [`mask-wearing`](https://universe.roboflow.com/joseph-nelson/mask-wearing) dari Roboflow Universe (2 kelas: `mask`, `no-mask`), yang cukup kecil: 105 gambar training, 29 gambar validasi. Untuk memperbaiki cakupan pada kasus nyata yang lebih sulit, kemudian digabungkan dataset publik kedua: [`Face-Mask-Detection`](https://universe.roboflow.com/hyssam-baccouche/face-mask-detection-djzcc) (848 gambar, 3 kelas: `with_mask`, `without_mask`, `mask_weared_incorrect`), mirror dari dataset Kaggle `andrewmvd/face-mask-detection`. Ketiga kelasnya dipetakan ulang ke skema 2-kelas dasar (`mask_weared_incorrect` → `no-mask`).

## Iterasi Model

Validasi dilakukan di 29 gambar held-out yang sama sepanjang proses, jadi angka di bawah ini bisa dibandingkan langsung antar versi.

| Versi | Backbone | imgsz Training | Data | mAP50 | mAP50-95 | Catatan |
|---|---|---|---|---|---|---|
| v1 | YOLOv8n | 640 | 105 gambar | 0.849 | 0.499 | Baseline |
| v2 | YOLOv8s | 960 | 105 gambar | 0.828 | 0.505 | Model lebih besar + resolusi training lebih tinggi |
| v3 | YOLOv8s | 960 | +763 gambar tambahan | 0.848 | 0.480 | Recall wajah sulit membaik, tapi muncul bias ke prediksi "mask" |
| v4 | YOLOv8s | 960 | data v3, porsi tambahan di-rebalance | 0.828 | 0.473 | Kelihatan bagus di atas kertas (precision no-mask 0.852), tapi pengujian nyata masih menunjukkan bias sesekali ke "mask" |
| v5 | YOLOv8s | 960 | data v4, keseimbangan kelas **total** di-rebalance (termasuk 105 gambar asli) | 0.732 | 0.428 | Skor validasi lebih rendah, tapi satu-satunya versi yang mengklasifikasi semua kasus wajah tanpa masker dengan benar |

mAP validasi v5 yang lebih rendah kelihatan seperti kemunduran di atas kertas. Pada praktiknya tidak — set validasi cuma 29 gambar dengan 20 instance "no-mask", jadi skornya sangat sensitif terhadap segelintir gambar, dan tidak mencerminkan performa di foto dunia nyata yang lebih beragam yang dipakai untuk pengecekan di bawah. Pengujian nyata di foto jalanan itu yang jadi penentu keputusan di sini, bukan tabel validasi.

## Keterbatasan yang Diketahui

- Dataset asli 105 gambar itu kecil dan condong ke contoh "mask" (573 instance mask vs 123 instance no-mask), yang jadi akar masalah bias yang dikejar sepanjang v3–v5.
- Sejumlah kombinasi pose/pencahayaan/hijab-dan-masker tertentu sesekali masih tidak terdeteksi bahkan di versi final — ini keterbatasan cakupan data, bukan sesuatu yang bisa sepenuhnya diperbaiki lewat tuning saat inferensi.
- Di confidence threshold rendah, model sesekali menghasilkan false positive pada objek bukan wajah (pernah teramati sekali: rumah kamera CCTV, di confidence rendah).
- Konfigurasi inferensi final: `conf=0.1`, `imgsz=1280`, `agnostic_nms=True`, plus dua filter pascaproses custom (filter kotak berukuran berlebihan dan filter kotak duplikat kelas-sama) yang didokumentasikan langsung di fungsi `detect_mask`.

## Hasil Deteksi (v5, konfigurasi final)

Foto jalanan nyata, tidak pernah dilihat model saat training, dijalankan lewat pipeline final:

![Hasil 1](assets/1.png)
![Hasil 2](assets/2.png)
![Hasil 3](assets/3.png)
![Hasil 4](assets/4.png)
![Hasil 5](assets/5.png)

## Demo Deployment

`Test_Model_Yolo_Mask.ipynb` menyertakan demo kecil yang menjalankan deteksi pada gambar yang diunggah dan mem-publish hasilnya (format JSON) ke broker MQTT (via broker publik [shiftr.io](https://www.shiftr.io/)), mensimulasikan bagaimana ini bisa masuk ke dashboard monitoring.

## Cara Menjalankan

Kedua notebook dibuat untuk dijalankan di Google Colab dengan runtime GPU.

1. Ambil API key gratis dari [Roboflow](https://app.roboflow.com/settings/api).
2. Di Colab, simpan sebagai secret bernama `ROBOFLOW_API_KEY` (ikon kunci di sidebar kiri), atau masukkan manual saat diminta.
3. Jalankan `Roboflow_Model_Yolo_Mask.ipynb` dari atas ke bawah (atau pakai cell "fast-forward" sebelum bagian v5 kalau sudah punya checkpoint `.pt` versi sebelumnya dan cuma mau mereproduksi tahap selanjutnya).
4. Bobot hasil training tersimpan ke `MyDrive/yolo-mask-detection/` di Google Drive.
5. Jalankan `Test_Model_Yolo_Mask.ipynb`, yang memuat model dari Drive dan memungkinkan upload gambar untuk dideteksi.

## Lisensi

MIT — lihat [LICENSE](LICENSE).
