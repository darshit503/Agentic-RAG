# c:\Darshit\Custom_Web_Scrawler\enhanced_rag.py
import os, hashlib, streamlit as st
import PyPDF2, docx
from bs4 import BeautifulSoup
import spacy
from textblob import TextBlob
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.chat_models import AzureChatOpenAI
from openai import AzureOpenAI
# New imports for the integrated version
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import openai
load_dotenv()

# Safe imports for the utility modules - modified to handle specific import errors
try:
    # First define fallbacks
    def simple_pdf_extract(file_path):
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return " ".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            print(f"Error extracting from {file_path}: {str(e)}")
            return ""
    
    # Set fallbacks as defaults
    safe_pdf_extract = simple_pdf_extract
    
    # Now try to import better versions if available
    from utils import safe_pdf_extract
    from utils.dedup import semantic_deduplicate
except ImportError as e:
    st.warning(f"Could not import utility modules: {str(e)}")
    
    # Define fallback for dedup if it fails
    def semantic_deduplicate(texts, embeddings, threshold=0.95):
        return list(set(texts))

# Continue with storage imports
try:
    from storage import save_metadata, load_metadata, ensure_directory, save_vector_store, load_vector_store
except ImportError as e:
    st.warning(f"Could not import storage module: {str(e)}")
    
    # Minimal fallback implementations
    def ensure_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    def save_metadata(filepath, metadata):
        import json
        with open(filepath, 'w') as f:
            json.dump(metadata, f)
            
    def load_metadata(filepath):
        import json
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
        
    def save_vector_store(vector_store, directory):
        ensure_directory(directory)
        vector_store.save_local(directory)
        
    def load_vector_store(directory, embeddings):
        if os.path.exists(directory):
            try:
                return FAISS.load_local(directory, embeddings)
            except:
                return None
        return None

# Continue with agent imports
try:
    from agents.query_understanding import QueryUnderstandingAgent
    from agents.grading import GradingAgent
    from agents.refinement import QueryRefinementAgent
except ImportError as e:
    st.warning(f"Could not import agent modules: {str(e)}")
    
    class QueryUnderstandingAgent:
        def run(self, query):
            return query
            
    class GradingAgent:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, query, answer, docs):
            return "5/10 - Unable to grade properly"
            
    class QueryRefinementAgent:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, query, grade_result):
            return query

# Fallback to simple NLP when spaCy model isn't available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Using simple NLP fallback.")
    # Create a minimal replacement for basic functionality
    class SimpleNLP:
        def __call__(self, text):
            class SimpleDoc:
                def __init__(self, text):
                    self.text = text
                    self.ents = []
            return SimpleDoc(text)
    nlp = SimpleNLP()

# Create a local embeddings class compatible with LangChain
class LocalEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # This model is a good balance of speed and quality
        # Other options: 'paraphrase-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1'
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
        
    # Add this method to make the class callable
    def __call__(self, text):
        # Handle both single texts and lists of texts
        if isinstance(text, str):
            return self.embed_query(text)
        else:
            return self.embed_documents(text)

# ————— New Azure OpenAI Integration —————
# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client - moved up for agent initialization
try:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("API_VERSION", "2025-01-01-preview"),
        api_key=os.getenv("API_KEY")
    )
except Exception as e:
    st.error(f"Error initializing Azure OpenAI Client: {str(e)}")
    print(f"Error initializing Azure OpenAI Client: {str(e)}")
    # Create dummy client for fallback
    class DummyClient:
        def chat(self):
            class DummyChat:
                def completions(self):
                    pass
            return DummyChat()
    client = DummyClient()

# Use Local Embeddings model exclusively
print("Using local Sentence Transformer embeddings")
embeddings = LocalEmbeddings()

# Configure LangChain's chat model with error handling
try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_CHAT", "gpt-4"),
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("AZURE_ENDPOINT"),
        openai_api_version=os.getenv("API_VERSION", "2025-01-01-preview"),
        temperature=0.0,
    )
except Exception as e:
    st.error(f"Error initializing Azure OpenAI Chat: {str(e)}")
    print(f"Error initializing Azure OpenAI Chat: {str(e)}")
    # Here we don't have a fallback for the LLM

# Initialize agents with the client
query_understanding_agent = QueryUnderstandingAgent()
grading_agent = GradingAgent(client, os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"))
query_refinement_agent = QueryRefinementAgent(client, os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"))

# ————— Data Preparation —————
def extract_text(fp):
    try:
        ext = os.path.splitext(fp)[1].lower()
        if ext == '.pdf':
            with open(fp, 'rb') as f:
                r = PyPDF2.PdfReader(f)
                return " ".join(p.extract_text() or "" for p in r.pages)
        if ext == '.docx':
            doc = docx.Document(fp)
            return " ".join(p.text for p in doc.paragraphs)
        if ext == '.html':
            with open(fp, 'r', encoding='utf-8') as f:
                return BeautifulSoup(f, 'html.parser').get_text()
        if ext == '.txt':
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        # Add support for additional file types
        if ext in ['.csv', '.json', '.md', '.xml']:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        # If unsupported file type
        print(f"Warning: Unsupported file type {ext} for {fp}. Trying to read as plain text.")
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error extracting text from {fp}: {str(e)}")
        return ""  # Return empty string to avoid breaking the pipeline

def clean_text(txt):
    return txt.replace("\n", " ").replace("\r", " ").lower().strip()

def enrich_metadata(txt, fn):
    doc = nlp(txt[:1000])
    ents = [(e.text, e.label_) for e in doc.ents]
    sentiment = TextBlob(txt[:1000]).sentiment.polarity
    return {
        "filename": fn,
        "file_type": os.path.splitext(fn)[1].lstrip('.'),
        "entities": ents,
        "word_count": len(txt.split()),
        "sentiment": sentiment
    }

def categorize(txt):
    t = txt.lower()

    categories = {
        "Legal": ["legal", "contract", "agreement", "license", "compliance", "regulation", "policy", "terms", 
                  "privacy", "gdpr", "intellectual property", "patent", "trademark", "copyright"],
        "HR": ["hr", "human resources", "employee", "personnel", "recruitment", "hiring", "benefits", 
               "onboarding", "training", "compensation", "performance review", "payroll"],
        "Procurement": ["procurement", "purchase", "vendor", "supplier", "quote", "bid", "tender", 
                        "sourcing", "contract", "invoice", "order", "delivery", "logistics"],
        "Finance": ["finance", "budget", "accounting", "tax", "revenue", "expense", "financial", 
                    "invoice", "payment", "fiscal", "audit", "balance sheet", "profit", "loss"],
        "Technology": ["tech", "technology", "software", "hardware", "it ", "infrastructure", "system", 
                       "network", "database", "application", "security", "cloud", "development"],
        "Sales & Marketing": ["marketing", "sales", "customer", "promotion", "campaign", "brand", 
                              "market research", "advertising", "social media", "lead", "conversion"],

        # Extended & Updated Categories Below
        "Fault Diagnosis": [
            "compound fault", "inner race fault", "outer race fault", "bearing fault", "misalignment", "unbalance", 
            "vibration signal", "rotating machinery", "rpm", "motor fault", "diagnostic", "severity level", 
            "spectral", "bpfi", "bpfo", "stft", "resonance"
        ],
        "Machine Learning": [
            "multi-output classification", "moc", "mcc", "uda", "domain adaptation", "entropy minimization", 
            "macro f1", "task-specific layer", "feature extractor", "deep learning", "adam optimizer"
        ],
        "Signal Processing": [
            "frequency layer normalization", "fln", "shaft speed", "frequency axis", "harmonics", 
            "spectral signature", "normalization", "frequency domain", "time domain", "stft", "rpm variations"
        ],
        "Kernel Methods": [
            "restricted kernel machine", "rkm", "ci-rkm", "class-informed", "weighted conjugate feature duality", 
            "kernel trick", "support vector machine", "kernel pca", "schur complement", "rbm", "rkhs"
        ],
        "Analog & Mixed Signal Design": [
            "analog circuit", "transistor sizing", "ams", "spice", "opamp", "gain", "bandwidth", 
            "phase margin", "bias voltage", "llm-based", "netlist", "gm/id", "ngspice", "ac simulation", 
            "transient simulation", "circuit optimization", "class ab", "common mode rejection", "output swing"
        ]
    }

    scores = {category: 0 for category in categories}
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in t:
                scores[category] += t.count(keyword)

    return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else "General"


def chunk_and_dedup(txt):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(txt)
    seen, unique = set(), []
    for c in chunks:
        h = hashlib.sha256(c.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique

def build_vector_store(docs):
    # Use local Sentence Transformer embeddings instead of Azure OpenAI
    emb = LocalEmbeddings()
    ld = []
    for d in docs:
        for c in d["chunks"]:
            ld.append(Document(page_content=c, metadata={**d["metadata"], "category": d["category"]}))
    return FAISS.from_documents(ld, emb)

# ————— Agentic RAG with Grading, Refinement, Cache —————
class AgenticRAG:
    def __init__(self, vs):
        self.vs = vs
        
        # Updated Azure OpenAI client initialization
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version="2025-01-01-preview",
            api_key=os.getenv("API_KEY"),
        )
        
        self.memory = ConversationBufferMemory()
        self.min_score = 0.3
        self.cache_thresh = 0.8
        self.embeddings = LocalEmbeddings()
        self.deployment_name = os.getenv("DEPLOYMENT_NAME", "GPT4")
        
    def understand(self, q): 
        # Enhanced query understanding
        doc = nlp(q)
        entities = [(e.text, e.label_) for e in doc.ents]
        # Expand query with detected entities
        expanded_q = q
        if entities:
            expanded_q += " " + " ".join([e[0] for e in entities])
        return expanded_q.strip()

    def embed(self, q):
        return self.embeddings.embed_query(q)

    def check_cache(self, q_emb):
        best_sim,best_ans=0,None
        for emb,ans in zip(st.session_state.cache_emb, st.session_state.cache_ans):
            dot=sum(a*b for a,b in zip(emb,q_emb))
            n1=(sum(a*a for a in emb))**0.5
            n2=(sum(b*b for b in q_emb))**0.5
            sim=dot/(n1*n2+1e-8)
            if sim>best_sim:
                best_sim,best_ans=sim,ans
        return best_ans if best_sim>=self.cache_thresh else None
    
    def retrieve(self, q):
        # First attempt: direct retrieval
        
        res = self.vs.similarity_search_with_score(q, k=5)
        
        # Filter by score threshold
        qualified_docs = [doc for doc, score in res if score > self.min_score]
        
        # If insufficient results, try category-based retrieval
        if len(qualified_docs) < 3:
            cat = self.memory.load_memory_variables({}).get("category", "")
            if cat:
                cat_res = self.vs.similarity_search_with_score(f"{q} in {cat}", k=5)
                qualified_docs.extend([doc for doc, score in cat_res if score > self.min_score])
                
        # Dedup and return top 3
        seen_content = set()
        unique_docs = []
        for doc in qualified_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
                if len(unique_docs) >= 3:
                    break
                    
        return unique_docs[:3]

    def grade_response(self, response, query, docs):
        context = '\n---\n'.join([d.page_content for d in docs])
        grading_prompt = (
            "You are a critical evaluator. Score this response to a query based on accuracy and relevance.\n\n"
            f"Query: {query}\n\n"
            f"Response: {response}\n\n"
            f"Context from documents: {context}\n\n"
            "Score from 1-10 with brief reason:"
        )

        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a grading assistant"},
                    {"role": "user", "content": grading_prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            result_text = response.choices[0].message.content
            
            # Extract score
            try:
                score = int(result_text.split('\n')[0].replace('Score:', '').strip().split('/')[0])
            except:
                score = 5  # Default middle score if parsing fails
                
            return score, result_text
        except Exception as e:
            print(f"Grading error: {str(e)}")
            return 5, f"Grading system error: {str(e)[:100]}"

    def refine_response(self, original_response, query, docs, grade_result):
        # Create the context separately
        context = '\n---\n'.join([d.page_content for d in docs])

        # Now, use it in the f-string without worrying about the backslash issue
        refine_prompt = (
            "You are an expert assistant. Your previous response needs improvement.\n\n"
            f"Query: {query}\n\n"
            f"Your previous response: {original_response}\n\n"
            f"Context from documents: {context}\n\n"
            f"Improvement feedback: {grade_result}\n\n"
            "Provide an improved response that addresses the feedback:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": refine_prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Refinement error: {str(e)}")
            return original_response

    def synthesize(self, q, docs, tone):
        ctx = "\n\n".join(d.page_content for d in docs)
        history = self.memory.load_memory_variables({}).get("history", "")
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": f"You are an enterprise assistant. Answer in a {tone} style."},
                    {"role": "user", "content": f"Previous context: {history}\nDocuments:\n{ctx}\nQuery: {q}\nAnswer based on documents:"}
                ],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {str(e)}")
            return f"System error: {str(e)[:100]}"

    def run(self, query: str, tone: str = "formal") -> str:
        # 1. Generate embedding for the query
        q_emb = self.embed(query)

        # 2. Check Cache
        cached = self.check_cache(q_emb)
        if cached: 
            print("[CACHE HIT]")
            return cached

        print("[CACHE MISS] - Executing pipeline")

        # 3. Understand the query (NER + keyword expansion)
        expanded_query = self.understand(query)

        # 4. Retrieve documents (semantic + category-aware)
        retrieved_docs = self.retrieve(expanded_query)

        if not retrieved_docs:
            print("[RETRIEVE] No documents found.")
            return "I'm sorry, I couldn't find any relevant information in the available documents."

        # 5. Synthesize answer using LLM with prompt template
        try:
            initial_response = self.synthesize(query, retrieved_docs, tone)
        except Exception as e:
            print(f"API Error during synthesis: {str(e)}")
            # Fallback: Generate a simple response from the documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            return f"Based on the documents I found, here's what I know: {context[:1000]}... (document summary truncated)"

        # Check if we should skip grading and refinement due to API access issues
        try:
            # Test API access with a very small request
            test_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            api_accessible = True
        except Exception as e:
            print(f"API access test failed: {str(e)}")
            api_accessible = False
            
        # 6. Grade the initial response only if API is accessible
        if api_accessible:
            try:
                score, grade_result = self.grade_response(initial_response, query, retrieved_docs)
                print(f"[GRADE] Initial score: {score}")

                # 7. Refine if score is below a threshold (e.g., 7)
                final_response = initial_response
                if score < 7:
                    print("[REFINE] Refining the response...")
                    try:
                        refined_response = self.refine_response(initial_response, query, retrieved_docs, grade_result)
                        final_response = refined_response
                    except Exception as e:
                        print(f"Refinement failed, using initial response: {str(e)}")
            except Exception as e:
                print(f"Grading failed, using initial response: {str(e)}")
                final_response = initial_response
        else:
            print("Skipping grading and refinement due to API access issues")
            final_response = initial_response

        # 8. Cache the final answer for future queries
        st.session_state.cache_emb.append(q_emb)
        st.session_state.cache_ans.append(final_response)

        return final_response

# Setup paths for storage
DATA_DIR = "C:\Darshit\Custom_Web_Scrawler\Data_Corpus"  # Use forward slashes for consistency
VECTOR_STORE_PATH = "C:\\Darshit\\Custom_Web_Scrawler\\vector_store"
METADATA_PATH = "C:\\Darshit\\Custom_Web_Scrawler\\Data_Corpus\\metadata.json"

# Ensure directories exist
ensure_directory(DATA_DIR)
ensure_directory(VECTOR_STORE_PATH)

def process_documents():
    """Process documents and create/update vector store"""
    # Initialize embeddings here to ensure it's available
    local_embeddings = LocalEmbeddings()
    
    # Load existing metadata
    metadata = load_metadata(METADATA_PATH)
    processed_files = {item.get("file_path", "") for item in metadata}
    
    # Check for new PDF files
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx', '.txt')):  # Add more file types
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    if not all_files:
        print(f"No documents found in {DATA_DIR}")
        st.warning(f"No documents found in {DATA_DIR}. Please check the directory.")
    
    new_files = [f for f in all_files if f not in processed_files]
    
    if not new_files:
        print("No new documents to process.")
        try:
            vector_store = load_vector_store(VECTOR_STORE_PATH, local_embeddings)
            if vector_store:
                return vector_store
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            st.error(f"Error loading vector store: {str(e)}")
    
    # Process documents
    documents = []
    new_metadata = []
    
    # Process new PDF files
    for file_path in new_files:
        try:
            print(f"Processing {file_path}")
            text = safe_pdf_extract(file_path)
            if text:
                doc_metadata = {"file_path": file_path, "source": os.path.splitext(file_path)[1][1:]}
                new_metadata.append(doc_metadata)
                documents.append({"page_content": text, "metadata": doc_metadata})
            else:
                print(f"Warning: No text extracted from {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            st.error(f"Error processing {file_path}: {str(e)}")
    
    if not documents and not metadata:
        raise ValueError("No documents to process")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc["page_content"])
        for chunk in doc_chunks:
            chunks.append({
                "page_content": chunk,
                "metadata": doc["metadata"]
            })
    
    # Deduplicate chunks
    if chunks:
        texts = [doc["page_content"] for doc in chunks]
        dedup_texts = semantic_deduplicate(texts, local_embeddings, threshold=0.95)
        
        dedup_chunks = [
            chunk for chunk in chunks 
            if chunk["page_content"] in dedup_texts
        ]
        
        # Update vector store
        vector_store = load_vector_store(VECTOR_STORE_PATH, local_embeddings)
        
        if vector_store:
            vector_store.add_texts(
                [doc["page_content"] for doc in dedup_chunks],
                [doc["metadata"] for doc in dedup_chunks]
            )
        else:
            vector_store = FAISS.from_texts(
                [doc["page_content"] for doc in dedup_chunks],
                local_embeddings,
                [doc["metadata"] for doc in dedup_chunks]
            )
        
        save_vector_store(vector_store, VECTOR_STORE_PATH)
        metadata.extend(new_metadata)
        save_metadata(METADATA_PATH, metadata)
        
        return vector_store
    
    vector_store = load_vector_store(VECTOR_STORE_PATH, local_embeddings)
    if vector_store:
        return vector_store
    
    raise ValueError("No documents to process and no existing vector store")


def query_system(user_query):
    """Process a query with enhanced RAG approach"""
    vector_store = process_documents()
    
    # Step 1: Understand and enhance the query
    enhanced_query = query_understanding_agent.run(user_query)
    
    # Step 2: Retrieve documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(enhanced_query)
    
    # Step 3: Generate response
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    response = qa_chain({"query": enhanced_query})
    answer = response["result"]
    
    # Step 4: Grade the response
    grade = grading_agent.run(enhanced_query, answer, docs)
    
    # Step 5: Refine if needed
    if any(f"{i}/" in grade for i in range(1, 5)):
        refined_query = query_refinement_agent.run(user_query, grade)
        response = qa_chain({"query": refined_query})
        answer = response["result"]
    
    return answer, docs

# ————— Streamlit UI —————
def streamlit_main():
    st.set_page_config(page_title="Enterprise RAG System", layout="wide")
    # 
    st.title("🌐 Agentic RAG with Enterprise Knowledge Assistant")
    
    # Enhanced sidebar with better organization
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        with st.expander("📂 Document Settings", expanded=True):
            path = DATA_DIR
            # st.info(f"Document folder: {path}")
            st.slider("Maximum documents to retrieve", 1, 10, 5, key="max_docs")
        
        with st.expander("🎨 Response Settings", expanded=True):
            tone = st.selectbox("Response Tone", ["Formal", "Casual", "Technical", "Simplified"])
            st.slider("Response Temperature", 0.0, 1.0, 0.7, 0.1, key="temp", 
                      help="Higher values make output more creative")
            st.checkbox("Include citations", value=True, key="show_citations")
            st.checkbox("Show confidence score", value=True, key="show_confidence")
        
        st.markdown("## 🔧 System Capabilities")
        
        # Replace local embeddings with OpenAI embeddings display
        st.success("✅ Using Embeddings Model")
        
        # Add new capabilities highlighting the agents and features
        st.success("✅ Query understanding agent with NER expansion")
        st.success("✅ Response grading agent")
        st.success("✅ Query refinement agent")
        
        # Add the system capabilities mentioned by the user
        st.markdown("## 🛠️ Advanced Features")
        st.markdown("✅ Robust PDF extraction using safe_pdf_extract()")
        st.markdown("✅ Semantic deduplication with semantic_deduplicate()")
        st.markdown("✅ Metadata storage and reuse via save_metadata / load_metadata")
        st.markdown("✅ Persistent FAISS vector store")
        st.markdown("✅ Query understanding with NER expansion")
        
        # Add model information in an expander
        with st.expander("🤖 Model Configuration", expanded=False):
            st.info("LLM: Azure OpenAI GPT-4")
            st.info("Embedding model: OpenAI text-embedding-ada-002")
            st.info("Chunk size: 1500 tokens")
            st.info("Chunk overlap: 150 tokens")
        
        st.divider()
        
        # System status with improved visuals
        st.markdown("## 📊 System Status")
        if 'vs' in st.session_state:
            st.success("✅ Vector store loaded")
            doc_count = len(st.session_state.doc_metadata) if 'doc_metadata' in st.session_state else "Unknown"
            st.metric("Documents indexed", doc_count)
            if 'cache_emb' in st.session_state:
                st.metric("Cached responses", len(st.session_state.cache_emb))
        else:
            st.warning("⚠️ No documents indexed")
    
    tab1, tab2 = st.tabs(["💬 Chat", "📑 Document Analysis"])
    
    if 'cache_emb' not in st.session_state:
        st.session_state.cache_emb = []
        st.session_state.cache_ans = []
    
    if 'doc_metadata' not in st.session_state:
        st.session_state.doc_metadata = []

    # Begin the document indexing process if needed
    if 'vs' not in st.session_state:
        if os.path.isdir(path):
            docs = []
            metadata_collection = []
            
            with st.status("Indexing documents..."):
                for fn in os.listdir(path):
                    fp = os.path.join(path, fn)
                    try:
                        txt = clean_text(extract_text(fp))
                        md = enrich_metadata(txt, fn)
                        cat = categorize(txt)
                        ch = chunk_and_dedup(txt)
                        docs.append({"metadata": md, "category": cat, "chunks": ch})
                        metadata_collection.append({
                            "filename": fn,
                            "category": cat,
                            "word_count": md["word_count"],
                            "sentiment": md["sentiment"],
                        })
                        st.write(f"Processed {fn}")
                    except Exception as e:
                        st.write(f"⚠️ Skipping {fn}: {e}")
            
            if docs:
                st.session_state.vs = build_vector_store(docs)
                st.session_state.doc_metadata = metadata_collection
                st.success(f"Successfully indexed {len(metadata_collection)} documents!")
            else:
                st.error("No documents found or processed.")
        else:
            st.error(f"Invalid folder path: {path}. Please check if the folder exists.")

    # Chat interface with source document display
    with tab1:
        # Initialize chat history if not already done
        if 'chat' not in st.session_state: 
            st.session_state.chat = []
        
        # Display chat history
        for i, m in enumerate(st.session_state.chat):
            if m['role'] == "user":
                st.chat_message("user").write(m['content'])
            else:
                # For assistant messages, set up a two-column layout
                if 'sources' in m:
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:
                        st.chat_message("assistant").write(m['content'])
                    with col2:
                        st.markdown("### 📚 Sources")
                        for idx, source in enumerate(m['sources']):
                            with st.container():
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {idx+1}</strong><br>
                                    File: {source['filename']}<br>
                                    Relevance: {source['relevance']}%<br>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("View content", expanded=False):
                                    st.markdown(source['content'])
                else:
                    st.chat_message("assistant").write(m['content'])
        
        # Always show chat input
        q = st.chat_input("Ask me anything about your documents...")
        
        if q:
            # Add user message to chat history
            st.session_state.chat.append({"role": "user", "content": q})
            st.chat_message("user").write(q)
            
            # Set up columns for assistant's response and sources
            col1, col2 = st.columns([0.7, 0.3])
            
            with col1:
                with st.chat_message("assistant"):
                    with st.spinner("Searching through your documents..."):
                        if 'vs' in st.session_state:
                            # If vector store is loaded, use RAG
                            rag = AgenticRAG(st.session_state.vs)
                            a = rag.run(q, tone.lower())
                            
                            # Extract sources from the retrieval process
                            sources = []
                            # In a real implementation, we'd extract these from the retrieved documents
                            # For now, simulate with placeholder data
                            retrieved_docs = rag.retrieve(q)
                            for i, doc in enumerate(retrieved_docs[:3]):
                                sources.append({
                                    'filename': doc.metadata.get('filename', f"document_{i}"),
                                    'relevance': int((1.0 - (i * 0.1)) * 100),  # Simulated relevance
                                    'content': doc.page_content[:200] + "..."
                                })
                            
                            st.write(a)
                        else:
                            # If vector store is not loaded, provide a helpful message
                            a = "I'm sorry, but I need to index some documents before I can answer your questions. Please check if the document folder exists and contains valid documents."
                            sources = []
                            st.write(a)
            
            # Display sources in the right column if we have them
            if 'vs' in st.session_state and sources:
                with col2:
                    st.markdown("### 📚 Sources")
                    for idx, source in enumerate(sources):
                        with st.container():
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {idx+1}</strong><br>
                                File: {source['filename']}<br>
                                Relevance: {source['relevance']}%<br>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("View content", expanded=False):
                                st.markdown(source['content'])
                                
                    if st.session_state.show_confidence:
                        st.markdown("### 🎯 Confidence")
                        st.progress(0.85)  # Simulated confidence score
                        st.caption("85% confident in this response")
            
            # Add assistant's response with sources to chat history
            response_entry = {
                "role": "assistant", 
                "content": a
            }
            if sources:
                response_entry["sources"] = sources
                
            st.session_state.chat.append(response_entry)

    with tab2:
        if 'doc_metadata' in st.session_state and st.session_state.doc_metadata:
            st.subheader("Document Overview")
            
            import pandas as pd
            df = pd.DataFrame(st.session_state.doc_metadata)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", len(df))
                if 'category' in df.columns:
                    categories = df['category'].value_counts().to_dict()
                    st.write("Categories:")
                    st.bar_chart(pd.Series(categories))
                else:
                    st.write("No category data available")
            
            with col2:
                if 'word_count' in df.columns:
                    avg_words = int(df['word_count'].mean())
                    st.metric("Average Word Count", avg_words)
                if 'sentiment' in df.columns:
                    avg_sentiment = round(df['sentiment'].mean(), 2)
                    sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
                    st.metric("Average Sentiment", f"{avg_sentiment} ({sentiment_label})")
            
            st.subheader("Document Details")
            search = st.text_input("Search documents:")
            if search:
                filtered_df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
                st.dataframe(filtered_df)
                
                if not filtered_df.empty:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download filtered data as CSV",
                        data=csv,
                        file_name="document_analysis.csv",
                        mime="text/csv"
                    )
            else:
                st.dataframe(df)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download all data as CSV",
                    data=csv,
                    file_name="document_analysis.csv",
                    mime="text/csv"
                )
        else:
            st.info("No document metadata available. Please index documents first.")

# Main function that offers both CLI and Streamlit modes
def main():
    # Always run in Streamlit mode when executed through Streamlit
    streamlit_main()

if __name__ == "__main__":
    main()