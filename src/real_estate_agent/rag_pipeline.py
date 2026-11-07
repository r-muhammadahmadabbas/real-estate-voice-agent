"""
RAG Pipeline for Real Estate Voice Agent
Uses ChromaDB (free, local) for vector storage
"""
import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging

from real_estate_agent.config import CHROMA_PATH, FAQ_PATH, PROPERTIES_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstateRAG:
    def __init__(self, chroma_path: str = None):
        self.chroma_path = str(chroma_path or CHROMA_PATH)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = None
        self.faq_df = None
        self.properties_df = None
        
    def load_data(self):
        """Load FAQ and property data"""
        faq_path = FAQ_PATH
        prop_path = PROPERTIES_PATH
        
        self.faq_df = pd.read_csv(faq_path)
        self.properties_df = pd.read_csv(prop_path)
        logger.info(f"Loaded {len(self.faq_df)} FAQs and {len(self.properties_df)} properties")
        
    def initialize_vector_store(self):
        """Initialize ChromaDB collection and embed FAQs"""
        self.collection = self.client.get_or_create_collection(
            name="real_estate_knowledge",
            metadata={"description": "Real estate FAQs and property descriptions"}
        )
        
        # Check if already populated
        if self.collection.count() > 0:
            logger.info(f"Vector store already has {self.collection.count()} documents")
            return
            
        # Embed FAQs
        self._embed_faqs()
        logger.info(f"Embedded {self.collection.count()} documents")
        
    def _embed_faqs(self):
        """Embed FAQ documents using simple text embedding (placeholder for real embedding)"""
        # For demo purposes, we'll use a simple keyword-based retrieval
        # In production, use sentence-transformers or OpenAI embeddings
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in self.faq_df.iterrows():
            doc_text = f"Q: {row['question']}\nA: {row['answer']}"
            documents.append(doc_text)
            metadatas.append({
                "category": row['category'],
                "source": "faq"
            })
            ids.append(f"faq_{row['id']}")
            
        # Add property descriptions
        for idx, row in self.properties_df.iterrows():
            if row['status'] == 'available':
                desc = f"{row['title']} in {row['area']}, {row['city']}. Price: {row['price']:,} PKR. "
                desc += f"Area: {row['area_sqft']} sqft. Type: {row['type']}. "
                desc += f"Amenities: {', '.join(eval(row['amenities'])) if row['amenities'] else 'None'}. "
                desc += f"Payment: {eval(row['payment_plans'])[0]['down']}% down, {eval(row['payment_plans'])[0]['installments']} installments."
                
                documents.append(desc)
                metadatas.append({
                    "property_id": row['id'],
                    "city": row['city'],
                    "area": row['area'],
                    "price": row['price'],
                    "type": row['type'],
                    "source": "property"
                })
                ids.append(f"prop_{row['id']}")
                
        # Simple keyword-based "embedding" - store text directly
        # In production, use: from sentence_transformers import SentenceTransformer
        # embeddings = model.encode(documents)
        # self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        
        # For demo, use chroma's default embedding (will use all-MiniLM-L6-v2)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search the vector store"""
        if not self.collection:
            self.initialize_vector_store()
            
        # Apply filters if provided
        where = filter_dict if filter_dict else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where
        )
        
        results_list = []
        for i in range(len(results['documents'][0])):
            results_list.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0,
                'id': results['ids'][0][i]
            })
            
        return results_list
    
    def search_properties(self, query: str, city: str = None, max_price: int = None, 
                          min_bedrooms: int = None, top_k: int = 5) -> List[Dict]:
        """Search properties with structured filters"""
        filter_dict = {"source": "property"}
        if city:
            filter_dict["city"] = city
            
        results = self.search(query, top_k=top_k, filter_dict=filter_dict)
        
        # Apply additional filters
        filtered = []
        for r in results:
            meta = r['metadata']
            if max_price and meta.get('price', 0) > max_price:
                continue
            if min_bedrooms and meta.get('bedrooms', 0) < min_bedrooms:
                continue
            filtered.append(r)
            
        return filtered[:top_k]
    
    def get_property_details(self, property_id: str) -> Optional[Dict]:
        """Get full property details by ID"""
        prop = self.properties_df[self.properties_df['id'] == property_id]
        if prop.empty:
            return None
        return prop.iloc[0].to_dict()


# Simple keyword-based retriever (fallback if no embeddings)
class KeywordRetriever:
    """Simple keyword-based retriever for demo without embeddings"""
    
    def __init__(self, faq_df: pd.DataFrame, prop_df: pd.DataFrame):
        self.faq_df = faq_df
        self.prop_df = prop_df
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score FAQs
        faq_scores = []
        for idx, row in self.faq_df.iterrows():
            text = (row['question'] + ' ' + row['answer']).lower()
            score = len(query_words & set(text.split()))
            if score > 0:
                faq_scores.append((score, {
                    'document': f"Q: {row['question']}\nA: {row['answer']}",
                    'metadata': {'category': row['category'], 'source': 'faq'},
                    'id': row['id']
                }))
        
        # Score properties
        prop_scores = []
        for idx, row in self.prop_df.iterrows():
            if row['status'] != 'available':
                continue
            text = f"{row['title']} {row['area']} {row['city']} {row['type']} {' '.join(eval(row['amenities']) if row['amenities'] else [])}".lower()
            score = len(query_words & set(text.split()))
            if score > 0:
                desc = f"{row['title']} in {row['area']}, {row['city']}. Price: {row['price']:,} PKR."
                prop_scores.append((score, {
                    'document': desc,
                    'metadata': {
                        'property_id': row['id'],
                        'city': row['city'],
                        'area': row['area'],
                        'price': row['price'],
                        'type': row['type'],
                        'source': 'property'
                    },
                    'id': row['id']
                }))
        
        # Combine and sort
        all_results = faq_scores + prop_scores
        all_results.sort(key=lambda x: x[0], reverse=True)
        
        return [r[1] for r in all_results[:top_k]]


if __name__ == "__main__":
    # Test the RAG pipeline
    rag = RealEstateRAG()
    rag.load_data()
    rag.initialize_vector_store()
    
    # Test searches
    test_queries = [
        "payment plan for DHA Phase 5",
        "5 marla plot in Lahore under 2 crore",
        "schools near DHA Phase 5",
        "commercial plot in DHA",
        "cheaper option than DHA Phase 5"
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = rag.search(q, top_k=3)
        for r in results:
            print(f"  - {r['document'][:100]}... (source: {r['metadata'].get('source')})")