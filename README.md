# 🏥 Medical Chatbot (DocBot)

## **Overview**
The **Medical Chatbot (DocBot)** is an AI-powered chatbot designed to assist users with medical queries by retrieving relevant information from a set of pre-loaded documents. It leverages **LangChain, FAISS, and Hugging Face's Mistral-7B-Instruct-v0.3** model to provide accurate responses.

Built using **Streamlit**, this chatbot allows users to engage in conversations where they can ask medical-related questions, and the chatbot responds based on the provided document context.

---

## **Features**
✅ **AI-Powered Conversations** – Uses **Mistral-7B-Instruct-v0.3** to generate responses.  
✅ **Document-Based Query Handling** – Extracts information from PDFs using **PyPDFLoader** and **FAISS** vector storage.  
✅ **Custom Prompting** – Ensures the chatbot stays within the provided medical context.  
✅ **Efficient Query Retrieval** – Implements **HuggingFaceEmbeddings** and **FAISS** for fast document search.  
✅ **Interactive UI** – Simple and user-friendly interface built using **Streamlit**.  

---

## **Tech Stack**
- **Python**
- **Streamlit** (for UI)
- **LangChain** (for conversational AI)
- **FAISS** (for vector storage)
- **Hugging Face Embeddings** (for document understanding)
- **Mistral-7B-Instruct-v0.3** (for LLM-based responses)
- **PyPDFLoader** (for PDF processing)

---

## **Installation & Setup**
#### **1. Clone the Repository**
```bash
git clone https://github.com/abhilasha522004/Medical-Chatbot.git
cd medical-chatbot
```

#### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
Alternatively, if you're using Pipenv:
```bash
pipenv install
```

#### **3. Set Environment Variables**
Create a `.env` file and add:
```
HF_TOKEN=your_huggingface_token
```

#### **4. Run the Application**
```bash
streamlit run DocBot.py
```

---

## **Project Structure**
```
medical-chatbot/
│── data/                      # Folder for PDF documents
│── vectorstore/               # FAISS vector database
│── DocBot.py                  # Main chatbot application
│── memory.py                   # Handles document processing and vector embedding
│── connect_memory.py          # (Optional) Memory management for persistent chats
│── Pipfile & Pipfile.lock      # Dependency management
│── .gitignore                  # Ignored files
│── README.md                   # Project documentation
```

---

## **How It Works**
1. **Document Processing**  
   - Loads medical PDFs from the `data/` folder using **PyPDFLoader**.  
   - Splits text into chunks using **RecursiveCharacterTextSplitter**.  
   - Converts text into embeddings using **Hugging Face's all-MiniLM-L6-v2**.  
   - Stores embeddings in a **FAISS** vector database.

2. **User Interaction**
   - The chatbot interface allows users to input queries.  
   - It retrieves relevant information from the vector database.  
   - Uses **Mistral-7B-Instruct-v0.3** to generate responses while staying within the provided context.

---

## **Example Usage**
🚀 Run the chatbot and ask:  
❓ *"What are the symptoms of diabetes?"*  
💡 **Response**: (Extracted from the medical PDFs)  

---

## **Future Enhancements**
🔹 **Speech-to-Text Support**  
🔹 **Multilingual Support**  
🔹 **Integration with External Medical APIs**  
🔹 **Live Chat Support with Doctors**  

---


Let me know if you need any modifications! 🚀
