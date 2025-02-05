import os
import yaml
import time
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm

from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextRecall,
    LLMContextRecall,
    FactualCorrectness
)
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# Config & Env
load_dotenv('config/.env')

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

class RateLimiter:
    """Enforce Gemini API rate limits (15 RPM)"""
    def __init__(self):
        self.last_call = 0
        self.request_count = 0
        
    def check_limit(self):
        if self.request_count >= 15:
            elapsed = time.time() - self.last_call
            if elapsed < 60:
                sleep_time = 60 - elapsed
                time.sleep(sleep_time)
                self.request_count = 0
        self.last_call = time.time()
        self.request_count += 1

class RAGSystem:
    def __init__(self):
        self.milvus_client = MilvusClient(uri=config['paths']['milvus_db'])

        self.embedder = HuggingFaceEmbeddings(
            model_name=config['models']['embedding_model'],
            model_kwargs={'device': config['device']['type']}
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=config['google_genai']['model'],
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            max_output_tokens=5000
        )
        self.rate_limiter = RateLimiter()

    def retrieve(self, query: str) -> List[Document]:
        results = self.milvus_client.search(
            collection_name="ellie_docs",
            data=[self._embed_query(query)],
            limit=config['retrieval']['top_k'],
            output_fields=["text", "source", "headers"]
        )
        return [
            Document(
                page_content=res["entity"]["text"],
                metadata={
                    "source": res["entity"]["source"],
                    "headers": res["entity"]["headers"]
                }
            ) for res in results[0]
        ]

    def generate(self, query: str, context: str) -> str:
        self.rate_limiter.check_limit()
        prompt = f"""Answer based on this provided context:\n{context}\n\nQuestion: {query}"""
        return self.llm.invoke(prompt).content

    def _embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)

# Evaluation Dataset
TEST_CASES = [
        {
            "user_input": "How do I login to Ellie for the first time?",
            "reference": "paste your Ellie login URL (eg. myorganization.ellie.ai) in your browser's address bar. If you haven't received a first time login password, click on the 'Reset password' button and enter your email address. If you don't receive a password reset shortly after, make sure your login URL is correct.",
            "relevant_docs": ["Frequently-Asked-Questions"]
        },
        {
            "user_input": "How to get started in Ellie?",
            "reference": "Ellie runs in the cloud; no installation is required. Just open your favorite web browser and navigate to your organization's Ellie instance. Usually, the URL is in the form of https://yourorganization.ellie.ai/. There are two options for logging in: 1. Log in with your Ellie account 2. Log in with your SSO provider",
            "relevant_docs": ["Ellie-User-Guide-the-basics"]
        },
        {
            "user_input": "How to access models in ellie?",
            "reference": "Data models in Ellie can be accessed from the front page under the Models header.",
            "relevant_docs": ["accessing-models", "Ellie-User-Guide-the-basics"]
        },
        {
            "user_input": "What is the user management?",
            "reference": "Ellie has two ways to manage users: 1. Email-based accounts: this is the default option, where each user gets an Ellie account based on their email address. 2. SSO integration: on request to support@ellie.ai, a customer organization can integrate their Ellie user management with their internal SSO provider. The options currently available are Okta and Azure Active Directory.",
            "relevant_docs": ["user-management", "Ellie-User-Guide-the-basics"]
        },
        {
            "user_input": "what are the user roles and privileges?",
            "reference": "Ellie has four levels of users: Admin, Write, Contributor, and Read. 1. Read: can read all content (models, Glossary items) but can't modify anything 2. Contributor: can read all content, and can work in a personal folder, and can copy public assets into a personal folder. 3. Write: can modify existing content and create new content (models, Glossary items), except for Folders where this has been separately restricted to specific Write-access users 4. Admin: can modify all existing content and create new content (models, Glossary items) everywhere in a subdomain",
            "relevant_docs": ["user-roles", "privileges", "Ellie-User-Guide-the-basics"]
        },
        {
            "user_input": "explain ellie's folder structure?",
            "reference": "Ellie's folder structure organizes models and glossaries, allowing users to create folders and subfolders to categorize models based on projects, domains, or departments. Each folder can contain its own sub-glossary, which lets users define terms specific to the folder's focus while maintaining access to the organization-wide glossaries",
            "relevant_docs": ["folders-and-subglossaries", "Ellie-User-Guide-the-basics"]
        },
        {
            "user_input": "explain the key features of folders in ellie",
            "reference": "- Hierarchical Organization: Create nested folders to represent various domains, projects, or departments, facilitating intuitive navigation and management. - Dedicated Sub-glossaries: Each folder can have its own glossary, known as a sub-glossary, allowing for domain-specific definitions while maintaining access to the organization-wide glossaries. - Access Control: Set permissions at the folder level to control who can edit the contents, ensuring security and proper governance.",
            "relevant_docs": ["key-features-folders-in-ellie", "working-with-folders-and-glossaries"]
        },
        {
            "user_input": "What are the types of folders are there in ellie?",
            "reference": "- Organization Folder: is the central workspace in Ellie where all shared assets are stored and managed. It serves as the main repository accessible to the entire organization. - Subfolders: are nested folders within the Organization Folder or other subfolders, enabling a hierarchical structure that aligns with specific domains, projects, or departments. Each subfolder can have its own sub-glossary for domain-specific terms while still allowing access to organization-wide glossaries. Permissions are inherited from parent folders but can be customized as needed. By default each folder has its own glossary enabled. - Personal folder: is a private space for individual users to draft, experiment, and refine models before sharing them with the broader organization. Users can copy assets from public folders, work on them privately, and publish finalized versions to public folders. Additionally, assets in the Personal Folder can be shared with admins or write-access users for review and publishing on the user's behalf.",
            "relevant_docs": ["types-of-folders", "working-with-folders-and-glossaries"]
        },
        {
            "user_input": "What is foreign entity?",
            "reference": "An entity borrowed from an other glossary is called a foreign entity. A foreign entity is not a copy, it is a reference to an entity from a foreign glossary. As a consequence: - If a foreign entity is modified, the changes are directly brought to the original entity. - If an entity is modified, all foreign usage of that entity will be updated.",
            "relevant_docs": ["foreign-entities", "working-with-folders-and-glossaries"]
        }
    
]

def prepare_dataset_(rag_system: RAGSystem):
    """Generate the Eval dataset"""
    dataset = []
    for case in tqdm(TEST_CASES, desc="Preparing Evaluation Data"):
        retrieved_docs = rag_system.retrieve(case["user_input"])
        context = [d.page_content for d in retrieved_docs]
        response = rag_system.generate(case['user_input'], "\n".join(context))
        
        dataset.append({
            "user_input": case["user_input"],
            "retrieved_contexts": context,
            "response": response,
            "reference": case["reference"]
        })

    return EvaluationDataset.from_list(dataset)

if __name__ == "__main__":
    rag_system = RAGSystem()
    
    evaluation_dataset= prepare_dataset_(rag_system)
    
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            Faithfulness(),
            AnswerCorrectness(),
            ContextRecall(),
            LLMContextRecall(),
            FactualCorrectness()
        ],
        llm=LangchainLLMWrapper(rag_system.llm),
        embeddings=rag_system.embedder
    )
    print('RAGAS results:', ragas_result)