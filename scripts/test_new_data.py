"""
Test if backend is retrieving from new data with metadata
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.chatbot_qdrant import chatbot_service


async def test_new_data():
    """Test queries that should return new data"""
    
    print("="*80)
    print("TESTING NEW DATA RETRIEVAL")
    print("="*80)
    
    # Initialize chatbot
    print("\nInitializing chatbot service...")
    await chatbot_service.initialize()
    print("✅ Initialized!")
    
    # Test queries that should match new data
    test_queries = [
        "Is the Qur'an a book of science?",
        "Can the Qur'an be used to prove modern scientific theories?",
        "What is the purpose of the Quran?",
    ]
    
    print(f"\n{'='*80}")
    print("TEST QUERIES")
    print(f"{'='*80}")
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        
        # Retrieve context
        context = chatbot_service.retrieve_context(query, top_k=3)
        
        if not context:
            print("❌ No results found!")
            continue
        
        for i, item in enumerate(context, 1):
            print(f"\n{i}. Score: {item['similarity']:.4f}")
            print(f"   Question: {item['instruction'][:80]}...")
            
            # Check for metadata
            has_metadata = 'channel_username' in item or 'video_id' in item
            
            if has_metadata:
                print(f"   ✅ HAS METADATA:")
                if 'channel_username' in item:
                    print(f"      📺 Channel: {item['channel_username']}")
                if 'video_id' in item:
                    print(f"      🎥 Video: https://youtube.com/watch?v={item['video_id']}")
                if 'source' in item:
                    print(f"      📁 Source: {item['source']}")
            else:
                print(f"   ❌ NO METADATA (from old data)")
                if 'source' in item:
                    print(f"      📁 Source: {item['source']}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Get stats
    stats = chatbot_service.get_stats()
    print(f"\nCurrent Configuration:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Vector DB: {stats['vector_db']}")
    
    # Check collection name
    from app.core.config import settings
    print(f"\nActive Collection: {settings.COLLECTION_NAME}")
    print(f"Data Path: {settings.DATA_PATH}")
    
    if settings.COLLECTION_NAME == "instructions_v2":
        print("\n✅ Using unified database (instructions_v2)")
        print("   You should see metadata in results above!")
    else:
        print(f"\n⚠️  Using old database ({settings.COLLECTION_NAME})")
        print("   To switch to unified database:")
        print("   1. Add to .env: COLLECTION_NAME=instructions_v2")
        print("   2. Add to .env: DATA_PATH=data/data_merged.json")
        print("   3. Restart backend")
    
    print()


if __name__ == "__main__":
    asyncio.run(test_new_data())
