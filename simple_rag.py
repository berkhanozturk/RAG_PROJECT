from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Ortam değişkenlerini yükle
load_dotenv()

# API anahtarını al
api_key = os.getenv("OPENAI_API_KEY")

# API Key kontrolü
if not api_key:
    print("❌ Lütfen .env dosyasında OPENAI_API_KEY tanımlayın.")
    exit()

# 1. PDF'ten belge yükle
loader = PyPDFLoader("test_belge.pdf")
pages = loader.load()

# 2. Belgeyi ChromaDB içine vektörleştir
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma.from_documents(pages, embedding)

# 3. RAG Soru-Cevap zinciri kur
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=api_key),
    retriever=vectordb.as_retriever()
)

# 4. Kullanıcıdan gelen soruları al
print("\n🤖 RAG Bot'a hoş geldin! PDF'ine soru sorabilirsin ('exit', 'q', 'quit' ile çık).\n")
while True:
    question = input("Soru: ")
    if question.lower() in ["exit", "q", "quit"]:
        print("Görüşürüz!")
        break

    answer = qa_chain.run(question)
    print(f"Cevap: {answer}\n")
