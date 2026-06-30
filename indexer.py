import os
import shutil
import stat
from git import Repo
from sentence_transformers import SentenceTransformer
import chromadb

SKIP_DIRS = {".git", "node_modules", "venv", "__pycache__", "dist", "build",".next"}

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c",".go", ".rb", ".md", ".css", ".html"}

def on_rn_error(func,path,exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repo(repo_url):
    local_dir = "cloned_repo"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir, onerror=on_rn_error)
        
    print(f"Cloning repo into ./{local_dir}...")
    Repo.clone_from(repo_url, local_dir)
    return local_dir

def find_code_files(root_dir):
    valid_files = []
    
    #os.walk unpacks 3 things: curernt directory path, sub-folders, and files
    for root, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        
        for file in filenames:
            ext = os.path.splitext(file)[1]
            if ext in CODE_EXTENSIONS:
                full_path = os.path.join(root, file)
                valid_files.append(full_path)
                
    return valid_files

def read_lines(file_path):
    with  open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()
    
def chunk_files(file_path, chunk_size = 40, overlap = 5):
    lines = read_lines(file_path)
    chunks = []
    
    if not lines:
        return chunks
    step = chunk_size - overlap
    
    for i in range(0, len(lines), step):
        chunk_slice = lines[i : i + chunk_size]
        joined_text = "".join(chunk_slice)
        
        chunk_data = {
            "file" : file_path,
            "start_line" : i + 1,
            "end_line" : min(i + chunk_size, len(lines)),
            "text" : joined_text 
        }
        
        chunks.append(chunk_data)
        
        if i + chunk_size >= len(lines):
            break
        
    return chunks
        
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("code_chunks")

def embed_texts(texts):
    embeddings = model.encode(texts)
    return embeddings.tolist()

def store_chunks(chunks):
    texts = [c["text"] for c in chunks]
    metadatas = [{"file": c["file"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    embeddings = embed_texts(texts)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

if __name__ == "__main__":
    repo_url = "https://github.com/mashalareej92-bit/Nasa_Explorer"
    
    path = clone_repo(repo_url)
    files = find_code_files(path)
    
    print(f"\nFound {len(files)} files")
    
    all_chunks = []
    for f in files[:20]:
        all_chunks.extend(chunk_files(f))
        
    print(f"Created {len(all_chunks)} chunks total")

    store_chunks(all_chunks)
    print(f"Stored. Collection now has {collection.count()} items.")