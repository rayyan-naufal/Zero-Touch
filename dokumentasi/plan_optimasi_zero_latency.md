# Rencana Implementasi: Optimasi Zero-Latency FaceControl

Dokumen ini berisi rencana teknis komprehensif untuk mengeksekusi optimasi performa FaceControl pada percakapan baru (`/plan`).

---

## 1. Ringkasan Tujuan (Goal)
Mengubah aplikasi FaceControl dari prototype berlatensi tinggi (**250–350ms input lag, 14–18 FPS**) menjadi aplikasi berkecepatan tinggi dengan respons instan (**< 30ms latency, 30.0 FPS terkunci**) tanpa jitter menggunakan arsitektur **Zero-Latency Ingestion & One Euro Filter**.

---

## 2. Rincian Masalah & Titik Perbaikan (Root Causes)

| No | Modul / Lokasi | Masalah Utama | Solusi Teknis |
| :--- | :--- | :--- | :--- |
| 1 | `gui.py: _start_camera_stream` | DirectShow driver menimbun 3–5 frame buffer (lag bawaan 150ms). | Set `CAP_PROP_BUFFERSIZE = 1`, `640x480`, `MJPG`, `30 FPS`. |
| 2 | `gui.py: _camera_worker` | `time.sleep(0.01)` memperlambat loop kamera hingga 14 FPS. | Hapus artificial sleep sepenuhnya. |
| 3 | `tracker.py: __init__` | `refine_landmarks=True` memuat 478 landmark (iris) yang tidak dipakai (boros 40% CPU). | Set `refine_landmarks=False` (468 landmark cukup). |
| 4 | `tracker.py: process_frame` | Filter kursor menggunakan EMA statis ($\alpha=0.35$), kursor tertinggal 250ms saat gerak cepat. | Ganti dengan algoritma adaptif **One Euro Filter ($1€$ Filter)**. |
| 5 | `gui.py: _update_gui_loop` | Tkinter main thread menjalankan `cv2.resize` dan `CTkImage` per tick (memicu UI freeze). | Offload resize & convert ke worker thread; Tkinter hanya render *latest frame*. |
| 6 | `gui.py: _camera_worker` | `pyautogui.position()` di-import dan dipanggil di setiap frame bahkan di mode absolute. | Gunakan direct Win32 `GetCursorPos` hanya jika di mode relative. |
| 7 | `mouse_controller.py: _execute_action` | Double click menjalankan `time.sleep(0.05)` yang memblokir loop kamera. | Dispatch double click secara non-blocking. |

---

## 3. Rencana Perubahan Berkas (Proposed Code Changes)

### 3.1. `tracker.py`
1. Tambahkan kelas filter adaptif `OneEuroFilter`:
   ```python
   class OneEuroFilter:
       def __init__(self, t0: float, x0: float, dx0: float = 0.0, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
           self.min_cutoff = min_cutoff
           self.beta = beta
           self.d_cutoff = d_cutoff
           self.x_prev = x0
           self.dx_prev = dx0
           self.t_prev = t0

       def filter(self, t: float, x: float) -> float:
           dt = max(1e-4, t - self.t_prev)
           # Dynamic cutoff based on speed
           edx = (x - self.x_prev) / dt
           dx = self._smoothing_factor(dt, self.d_cutoff) * edx + (1 - self._smoothing_factor(dt, self.d_cutoff)) * self.dx_prev
           cutoff = self.min_cutoff + self.beta * abs(dx)
           a = self._smoothing_factor(dt, cutoff)
           x_hat = a * x + (1 - a) * self.x_prev
           self.x_prev = x_hat
           self.dx_prev = dx
           self.t_prev = t
           return x_hat
   ```
2. Ganti `refine_landmarks=True` menjadi `refine_landmarks=False`.
3. Inisialisasi `filter_x` dan `filter_y` untuk koordinat layar kursor.

### 3.2. `gui.py`
1. Konfigurasi kamera di `_start_camera_stream`:
   ```python
   self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
   self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
   self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   self.cap.set(cv2.CAP_PROP_FPS, 30)
   ```
2. Hapus `time.sleep(0.01)` di dalam `_camera_worker`.
3. Pindahkan konversi gambar:
   Lakukan `cv2.resize(frame, (480, 360))` dan konversi warna RGB di worker thread sebelum dimasukkan ke antrean queue GUI.
4. Di `_update_gui_loop`, ambil frame terakhir dengan pola drop-stale:
   ```python
   latest = None
   while not self.frame_queue.empty():
       latest = self.frame_queue.get_nowait()
   if latest is not None:
       # Update canvas sekali saja
   ```

### 3.3. `mouse_controller.py`
1. Hilangkan `time.sleep(0.05)` di bagian double click.
2. Optimalkan debouncing click event agar jeda klik tidak berlebihan saat gesture terdeteksi.

### 3.4. `test_facecontrol.py`
1. Tambahkan unit test untuk `OneEuroFilter` guna memverifikasi:
   - Stabilitas tinggi saat kecepatan rendah ($|v| \to 0$).
   - Latensi nol saat kecepatan tinggi ($|v| \gg 0$).
2. Pastikan seluruh 5 pengujian sebelumnya tetap lulus (100% PASS).

---

## 4. Panduan Melanjutkan di Conversation Baru

Saat membuka percakapan baru, Anda cukup memberikan instruksi:
```text
Lanjutkan eksekusi perbaikan sesuai plan di dokumentasi/plan_optimasi_zero_latency.md
```
Agen akan membaca rencana ini dan langsung melakukan perubahan kode serta verifikasi otomatis.
