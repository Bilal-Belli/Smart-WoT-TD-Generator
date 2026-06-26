import os
import json
import warnings
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LangChain imports
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Local embeddings for fast filtering
from sentence_transformers import SentenceTransformer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# 1. SMART DESCRIPTION GENERATOR (CACHED)
# =========================================================

class SmartDescriptionGenerator:
    """Generates and caches augmented descriptions for things"""
    def __init__(self, cache_file="smart_descriptions_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load cached descriptions from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def create_smart_description(self, thing: Dict) -> str:
        thing_id = str(thing.get('_id') or thing.get('id') or '')
        if thing_id in self.cache:
            return self.cache[thing_id]
        parts = []
        # Title and description
        if thing.get('title'):
            parts.append(f"Title: {thing['title']}")
        if thing.get('description'):
            parts.append(f"Description: {thing['description']}")

        # Properties: include name, description, type, unit
        if thing.get('properties'):
            properties = thing['properties']
            if isinstance(properties, dict):
                for name, info in properties.items():
                    parts.append(f"Property: {name}")
                    if isinstance(info, dict):
                        if info.get('description'):
                            parts.append(f"Property description: {info['description']}")
                        if info.get('type'):
                            parts.append(f"Property type: {info['type']}")
                        if info.get('unit'):
                            parts.append(f"Property unit: {info['unit']}")
            elif isinstance(properties, list):
                for prop in properties:
                    if isinstance(prop, dict):
                        name = prop.get('title') or prop.get('name') or ''
                        parts.append(f"Property: {name}")
                        if prop.get('description'):
                            parts.append(f"Property description: {prop['description']}")
                        if prop.get('type'):
                            parts.append(f"Property type: {prop['type']}")
                        if prop.get('unit'):
                            parts.append(f"Property unit: {prop['unit']}")

        # Actions: include name and description
        if thing.get('actions'):
            actions = thing['actions']
            if isinstance(actions, dict):
                for name, info in actions.items():
                    parts.append(f"Action: {name}")
                    if isinstance(info, dict) and info.get('description'):
                        parts.append(f"Action description: {info['description']}")
            elif isinstance(actions, list):
                for action in actions:
                    if isinstance(action, dict):
                        name = action.get('title') or action.get('name') or ''
                        parts.append(f"Action: {name}")
                        if action.get('description'):
                            parts.append(f"Action description: {action['description']}")

        # Events: include name and description
        if thing.get('events'):
            events = thing['events']
            if isinstance(events, dict):
                for name, info in events.items():
                    parts.append(f"Event: {name}")
                    if isinstance(info, dict) and info.get('description'):
                        parts.append(f"Event description: {info['description']}")
            elif isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        name = event.get('title') or event.get('name') or ''
                        parts.append(f"Event: {name}")
                        if event.get('description'):
                            parts.append(f"Event description: {event['description']}")

        # Security and base URL
        if thing.get('security'):
            parts.append(f"Security: {', '.join(map(str, thing['security']))}")
        if thing.get('base'):
            parts.append(f"Base URL: {thing['base']}")

        augmented = " | ".join(parts)
        self.cache[thing_id] = augmented
        self._save_cache()
        return augmented

    def get_batch_descriptions(self, things: List[Dict]) -> List[str]:
        """Get descriptions for multiple things"""
        return [self.create_smart_description(thing) for thing in things]

# =====================================
# 2. FAST FILTER USING LOCAL EMBEDDINGS
# =====================================

class FastThingFilter:
    """First-stage filter using lightweight local embeddings"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.smart_desc_gen = SmartDescriptionGenerator()
    
    def prefilter(self, query: str, things: List[Dict], threshold: float = 0.05, top_k: int = 20) -> List[Dict]:
        """
        Fast pre-filtering of things using local embeddings
        Returns only potentially relevant things for LLM processing
        """
        if not things:
            return []
        
        # Generate augmented descriptions for all things (uses cache)
        descriptions = self.smart_desc_gen.get_batch_descriptions(things)
        
        # Encode query once
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Encode all descriptions
        thing_embeddings = self.model.encode(descriptions, normalize_embeddings=True)
        
        # Calculate similarities (cosine with normalized vectors)
        similarities = np.dot(thing_embeddings, query_embedding)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Take top_k
        top_indices = sorted_indices[:top_k]
        
        filtered_things = [things[i] for i in top_indices]
        
        top_sim = similarities[top_indices[0]] if len(top_indices) > 0 else 0.0
        print(f"[STAGE 1] Filtered from {len(things)} to {len(filtered_things)} candidates (top similarity: {top_sim:.3f})")
        
        return filtered_things

# ==========================================
# 3. MONGO DB MANAGER WITH BATCH PROCESSING
# ==========================================

class MongoDBManager:
    """Handles MongoDB operations with batch processing"""
    
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="wot_repository", collection_name="thing_descriptions"):
        self.demo_mode = False
        try:
            self.client = MongoClient(connection_string)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            
            # Create indexes for faster queries
            self.collection.create_index("title")
            self.collection.create_index("description")
            self.collection.create_index([("title", "text"), ("description", "text")])
            
            print(f"Connected to MongoDB: {db_name}.{collection_name}")
            self.sync_with_json_file()
        except ConnectionFailure:
            print("Warning: Could not connect to MongoDB. Running in demo mode with sample data.")
            self._init_demo_mode()
    
    def _init_demo_mode(self):
        """Initialize in-memory storage for demo when MongoDB is not available"""
        self.demo_mode = True
        self.demo_data = {}
        self.sync_with_json_file()
        
    def sync_with_json_file(self):
        """Syncs MongoDB or demo mode data with local things-database.json file"""
        db_path = "things-database.json"
        if not os.path.exists(db_path):
            print(f"Local {db_path} not found. Skipping sync.")
            return
        
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                things = json.load(f)
            
            if not isinstance(things, list):
                print(f"Warning: {db_path} does not contain a list of things.")
                return

            if hasattr(self, 'demo_mode') and self.demo_mode:
                # Update in-memory demo data
                self.demo_data = {}
                for thing in things:
                    thing_id = thing.get('id') or thing.get('_id')
                    if thing_id:
                        doc = thing.copy()
                        doc['_id'] = str(thing_id)
                        doc['id'] = str(thing_id)
                        self.demo_data[str(thing_id)] = doc
                print(f"Loaded {len(self.demo_data)} things from local {db_path} into memory (demo mode).")
            else:
                # Sync with MongoDB
                print(f"Syncing MongoDB with {len(things)} things from local {db_path}...")
                active_ids = []
                for thing in things:
                    thing_id = thing.get('id') or thing.get('_id')
                    if not thing_id:
                        continue
                    active_ids.append(str(thing_id))
                    
                    # Make a copy of thing and set _id for MongoDB
                    mongo_doc = thing.copy()
                    mongo_doc['_id'] = str(thing_id)
                    mongo_doc['id'] = str(thing_id)
                    
                    # Upsert into MongoDB
                    self.collection.replace_one({"_id": str(thing_id)}, mongo_doc, upsert=True)
                
                # Clean up documents in MongoDB that are no longer in the JSON database
                delete_result = self.collection.delete_many({"_id": {"$nin": active_ids}})
                if delete_result.deleted_count > 0:
                    print(f"Deleted {delete_result.deleted_count} stale things from MongoDB.")
                print("Sync with MongoDB complete.")
        except Exception as e:
            print(f"Error syncing with JSON file: {e}")
    
    def get_all_things(self, batch_size: int = 1000):
        """Get all things with batching to avoid memory issues"""
        if hasattr(self, 'demo_mode') and self.demo_mode:
            return list(self.demo_data.values())
        
        things = []
        cursor = self.collection.find({}, batch_size=batch_size)
        for doc in cursor:
            things.append(doc)
        return things
    
    def get_thing_by_id(self, thing_id: str) -> Optional[Dict]:
        """Fast lookup by ID"""
        if hasattr(self, 'demo_mode') and self.demo_mode:
            return self.demo_data.get(thing_id)
        return self.collection.find_one({"_id": thing_id})

# ======================================
# 4. LLM ROUTER (ONE API CALL PER QUERY) 
# ======================================

from pydantic import BaseModel, Field

# 1. Define the rigid schema we want back from Gemini
class RouterDecision(BaseModel):
    matched_id: str = Field(
        description="The exact string ID of the matching candidate device (e.g., 'urn:dev:wot:weather-station-2'). If absolutely no candidate matches the query, return 'NOT_FOUND'."
    )

class LLMRouter:
    """Handles the final routing decision using Gemini (one API call)"""
    
    def __init__(self, gemini_api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=gemini_api_key
        )
        
        # Initialize the base LLM with 0 temperature for maximum precision
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=gemini_api_key
        )
        
        # 2. FORCE Gemini to output our strict schema using native tool-calling
        self.structured_llm = base_llm.with_structured_output(RouterDecision)
        self.vectorstore = None
        self.retriever = None
    
    def setup_vectorstore(self, things: List[Dict]):
        """Create vectorstore from filtered things (temporary, per query)"""
        if not things:
            return
        
        documents = []
        smart_gen = SmartDescriptionGenerator()
        
        for thing in things:
            augmented_desc = smart_gen.create_smart_description(thing)
            from langchain_core.documents import Document
            
            # Extract the real target ID
            thing_id = thing.get('_id') or thing.get('id') or thing.get('urn') or 'unknown'
            
            doc = Document(
                page_content=augmented_desc,
                metadata={'id': str(thing_id)} # Store the string version of the actual ID
            )
            documents.append(doc)
        
        import uuid
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            ids=[str(uuid.uuid4()) for _ in documents],
            persist_directory=None
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": min(5, len(documents))})
    
    def create_prompt(self) -> ChatPromptTemplate:
        """Create the routing prompt"""
        template = """
You are an expert Web of Things (WoT) database router.

Analyze the user's natural language query and review the candidate devices provided in the context.
Determine which device matches the user's goal best and select its exact ID.

Context Candidates:
{context}

User Query: {question}
"""
        return ChatPromptTemplate.from_template(template)
    
    # 3. CRITICAL CORRECTION: Inject the ID directly into the text Gemini reads!
    def format_docs(self, docs):
        formatted_blocks = []
        for doc in docs:
            real_id = doc.metadata.get('id')
            block = f"--- CANDIDATE DEVICE ---\nEXACT_ID_TO_RETURN: {real_id}\nDetails: {doc.page_content}"
            formatted_blocks.append(block)
        return "\n\n".join(formatted_blocks)
    
    def route(self, query: str, candidates: List[Dict]) -> str:
        """Route query to the best matching thing ID"""
        if not self.retriever:
            return "NOT_FOUND"
        
        # If only one candidate remains from Stage 1, bypass the LLM entirely
        if len(candidates) == 1:
            thing_id = candidates[0].get('_id') or candidates[0].get('id') or 'unknown'
            return str(thing_id)
        
        prompt = self.create_prompt()
        
        # Set up chain running through our strict pydantic binding schema
        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.structured_llm
        )
        
        try:
            # Result automatically outputs as our clean Pydantic object model
            decision = rag_chain.invoke(query)
            return decision.matched_id.strip()
        except Exception as e:
            print(f"  → LLM routing error: {e}")
            return "NOT_FOUND"

# ===================
# 5. MAIN APPLICATION
# ===================

class WoTThingRouter:
    """Main application orchestrating the two-stage filtering"""
    
    def __init__(self):
        # Initialize components
        self.mongodb = MongoDBManager()
        self.all_things = []
        self.last_db_mtime = 0
        self.fast_filter = FastThingFilter()
        self.llm_router = LLMRouter(os.environ.get("GEMINI_API_KEY"))
        
        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "total_llm_calls": 0,
            "cache_hits": 0
        }
        
        # Initial sync and load
        self.refresh_database()

    def refresh_database(self):
        """Check if things-database.json has changed and sync/reload if necessary"""
        db_path = "things-database.json"
        if os.path.exists(db_path):
            try:
                mtime = os.path.getmtime(db_path)
                if mtime > self.last_db_mtime:
                    # Sync MongoDB/demo data with JSON
                    self.mongodb.sync_with_json_file()
                    # Reload all things from MongoDB (or local storage)
                    self.all_things = self.mongodb.get_all_things()
                    self.last_db_mtime = mtime
                    print(f"Database reloaded/synced: {len(self.all_things)} things.")
            except Exception as e:
                print(f"Error checking/reloading database: {e}")
        else:
            # If JSON doesn't exist, load whatever is in MongoDB/demo
            if not self.all_things:
                self.all_things = self.mongodb.get_all_things()
    
    def process_query(self, user_query: str) -> Optional[Dict]:
        """
        Process a user query through two-stage filtering:
        1. Fast local embedding filter
        2. LLM routing (1 API call)
        """
        self.refresh_database()
        self.stats["total_queries"] += 1
        
        print("\n" + "=" * 70)
        print(f"Query #{self.stats['total_queries']}: {user_query}")
        print("=" * 70)
        
        # STAGE 1: Pre-filtering
        candidates = self.fast_filter.prefilter(
            user_query, 
            self.all_things,
            threshold=0.05,
            top_k=5
        )
        
        if not candidates:
            print("\nNo candidate matches found in stage 1 filtering.")
            return None
        
        # STAGE 2: LLM routing (API call)
        self.stats["total_llm_calls"] += 1
        
        # Setup temporary vectorstore with candidates only
        self.llm_router.setup_vectorstore(candidates)
        
        # Get the final decision (FIX: pass candidates to route)
        target_id = self.llm_router.route(user_query, candidates)
        print(f"[STAGE 2] RAG decision: {target_id}")
        
        if target_id == "NOT_FOUND":
            print("\nNo matching device found in the candidate set.")
            return None
        
        # Retrieve the full thing from MongoDB
        print("\n[RETRIEVAL] Fetching thing description")
        thing = self.mongodb.get_thing_by_id(target_id)
        
        if thing:
            return thing
        else:
            print(f"\nWarning: LLM returned ID '{target_id}' but not found in database")
            return None
    
    def display_thing(self, thing: Dict):
        
        display_thing = thing.copy()
        if "_id" in display_thing:
            display_thing["id"] = display_thing.pop("_id")
        
        print(json.dumps(display_thing, indent=2))

    def display_stats(self):
        """Display query statistics"""
        print("\n" + "=" * 40)
        print("STATISTICS")
        print("=" * 40)
        print(f"Total queries processed: {self.stats['total_queries']}")
        print(f"Total LLM API calls: {self.stats['total_llm_calls']}")
        print(f"API calls per query (average): {self.stats['total_llm_calls'] / max(1, self.stats['total_queries']):.1f}")
        print("=" * 40)
    
    def run_interactive(self):
        """Run interactive query loop"""
        
        while True:
            try:
                user_input = input("\nDescribe your desired device ('exit' to quit, 'stats' for statistics): ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                if user_input.lower() == 'stats':
                    self.display_stats()
                    continue
                
                if not user_input:
                    continue
                
                result = self.process_query(user_input)
                
                if result:
                    self.display_thing(result)
                else:
                    print("\nCould not find a matching device. Please try a different description.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self.display_stats()
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

# ========
# 6. MAIN
# ========

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        return
    router = WoTThingRouter()
    router.run_interactive()

if __name__ == "__main__":
    main()