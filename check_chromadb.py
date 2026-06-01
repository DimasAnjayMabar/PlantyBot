import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("konten_isi_raw")
print("Jumlah dokumen:", col.count())