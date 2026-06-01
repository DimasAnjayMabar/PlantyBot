# Jalankan script repair
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("konten_isi_raw")
except:
    pass
client.create_collection("konten_isi_raw")
print("Collection raw telah direset")