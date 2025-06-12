# Konfigurasi global untuk daftar kode saham yang akan diproses dalam pipeline
# Daftar ini dapat diubah sesuai kebutuhan, pastikan menggunakan kode saham dari BEI tanpa akhiran ".JK" 
# karena akhiran tersebut akan ditambahkan secara otomatis pada tahapan extract.

# List ticker saham
TICKERS = ['BBRI.JK', 'BMRI.JK', 'BBCA.JK', 'BBNI.JK','GOTO.JK','TLKM.JK', 'AAPL']