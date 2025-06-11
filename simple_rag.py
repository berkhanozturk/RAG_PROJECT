import os
from langchain.text_splitter import CharacterTextSplitter

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
    print("❌ 'pdfs/' klasöründe PDF yok.")
    exit()

print(f"🔍 {len(pdf_paths)} PDF bulundu. Yükleniyor...")

for path in pdf_paths:
    loader = PyPDFLoader(path)
    loaded_docs = loader.load()

    # 🔹 Her dosya için parçalama
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(loaded_docs)

    docs.extend(split_docs)

print(f"✅ Toplam {len(docs)} parçaya bölünmüş belge yüklendi.")

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

    # Cevap üretmeden önce chunk'ları görelim
    relevant = qa_chain.retriever.get_relevant_documents("kırmızı kablo nerede")
    for i, d in enumerate(relevant):
        print(f"\n🎯 Chunk {i+1}:\n{d.page_content}")


    answer = qa_chain.run(question)
    print(f"📎 Cevap: {answer}\n")
