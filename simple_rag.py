import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import glob

# ✅ OpenAI API anahtarını kontrol et
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY ortam değişkenini tanımla.")
    exit()

# ✅ Tüm PDF dosyalarını yükle
docs = []
pdf_paths = glob.glob("pdfs/*.pdf")

if not pdf_paths:
    print("❌ 'pdfs/' klasöründe hiç PDF yok.")
    exit()

print(f"🔍 {len(pdf_paths)} PDF bulundu. Yükleniyor...")

for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

print(f"✅ Toplam {len(docs)} sayfa yüklendi.")

# ✅ Belgeleri OpenAI ile vektörleştir ve Chroma'ya kaydet
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embedding)

# ✅ RAG soru-cevap zinciri kur
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever()
)

# ✅ Kullanıcıdan gelen sorular
print("\n🤖 Çoklu PDF RAG Bot'a hoş geldin! Soru sorabilirsin ('exit' yaz çık).\n")

while True:
    question = input("Soru: ")
    if question.lower() in ["exit", "q", "quit"]:
        print("👋 Görüşmek üzere!")
        break

    answer = qa_chain.run(question)
    print(f"📎 Cevap: {answer}\n")
