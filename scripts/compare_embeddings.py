"""
Compare embedding quality: GloVe vs sentence-transformers
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.chatbot_glove import ChatbotServiceGloVe
from app.services.chatbot import ChatbotService


async def test_embedding_quality():
    """Compare GloVe vs sentence-transformers"""
    
    print("="*80)
    print("EMBEDDING QUALITY COMPARISON")
    print("="*80)
    
    # Test questions
    test_questions = [
        "What is Islam?",
        "How do I pray?",
        "What is Ramadan?",
        "Tell me about fasting",
        "What is Hajj?",
        "What are the five pillars?",
        "How to perform ablution?",
        "What is Zakat?",
    ]
    
    print("\n🔤 Initializing GloVe service...")
    glove_service = ChatbotServiceGloVe()
    await glove_service.initialize()
    
    print("\n🤖 Initializing sentence-transformers service...")
    st_service = ChatbotService()
    await st_service.initialize()
    
    print("\n" + "="*80)
    print("TESTING QUERIES")
    print("="*80)
    
    for question in test_questions:
        print("\n" + "-"*80)
        print(f"📝 Question: {question}")
        print("-"*80)
        
        # Test with GloVe
        print("\n🔤 GloVe Results:")
        glove_context = await glove_service.retrieve_context(question, top_k=3)
        for i, ctx in enumerate(glove_context):
            print(f"  {i+1}. Score: {ctx['similarity']:.4f}")
            print(f"     {ctx['instruction'][:70]}...")
        
        # Test with sentence-transformers
        print("\n🤖 sentence-transformers Results:")
        st_context = st_service.retrieve_context(question, top_k=3)
        for i, ctx in enumerate(st_context):
            print(f"  {i+1}. Score: {ctx['similarity']:.4f}")
            print(f"     {ctx['instruction'][:70]}...")
        
        # Compare
        print("\n📊 Comparison:")
        glove_avg = sum(c['similarity'] for c in glove_context) / len(glove_context) if glove_context else 0
        st_avg = sum(c['similarity'] for c in st_context) / len(st_context) if st_context else 0
        
        print(f"  GloVe avg similarity: {glove_avg:.4f}")
        print(f"  sentence-transformers avg similarity: {st_avg:.4f}")
        print(f"  Difference: {abs(glove_avg - st_avg):.4f}")
        
        if st_avg > glove_avg:
            print(f"  ✅ sentence-transformers is {((st_avg - glove_avg) / glove_avg * 100):.1f}% better")
        else:
            print(f"  ✅ GloVe is {((glove_avg - st_avg) / st_avg * 100):.1f}% better")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n🔤 GloVe:")
    print(f"  - Embedding dimension: {glove_service.embedder.embedding_dim}")
    print(f"  - Vocabulary size: {len(glove_service.embedder.embeddings_dict)}")
    print(f"  - Package size: ~250 MB")
    print(f"  - Vercel compatible: ✅ Yes")
    
    print("\n🤖 sentence-transformers:")
    print(f"  - Model: all-MiniLM-L6-v2")
    print(f"  - Embedding dimension: 384")
    print(f"  - Package size: ~1.5 GB")
    print(f"  - Vercel compatible: ❌ No")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(test_embedding_quality())
