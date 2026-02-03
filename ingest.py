from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

# 1. 讀取所有 md 文件
docs = []
for path in Path(".").glob("*.md"):
    text = path.read_text(encoding="utf-8")
    docs.append({
        "content": text,
        "source": path.name
    })

print(f"Loaded {len(docs)} documents")

# 2. 切 chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

texts = []
metadatas = []

for doc in docs:
    chunks = splitter.split_text(doc["content"])
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({"source": doc["source"]})

print(f"Created {len(texts)} chunks")

# 3. 建立向量資料庫
embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.persist()

print("Vector store created successfully!")
