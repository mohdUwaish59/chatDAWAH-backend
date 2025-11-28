"""
Check which collections exist in Qdrant and their details
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from app.core.config import settings


def check_collections():
    """Check all collections in Qdrant"""
    
    print("="*80)
    print("QDRANT COLLECTIONS CHECK")
    print("="*80)
    
    if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
        print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        return
    
    print(f"\nConnecting to: {settings.QDRANT_URL}")
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    
    # Get all collections
    collections = qdrant_client.get_collections().collections
    
    print(f"\n📊 Found {len(collections)} collection(s):\n")
    
    for collection in collections:
        print(f"{'='*80}")
        print(f"Collection: {collection.name}")
        print(f"{'='*80}")
        
        # Get detailed info
        info = qdrant_client.get_collection(collection.name)
        
        print(f"  Points (items): {info.points_count}")
        print(f"  Vector size: {info.config.params.vectors.size} dimensions")
        print(f"  Distance metric: {info.config.params.vectors.distance}")
        print(f"  Status: {info.status}")
        
        # Check if this is the active collection
        if collection.name == settings.COLLECTION_NAME:
            print(f"  ✅ CURRENTLY ACTIVE (configured in .env)")
        else:
            print(f"  ⚪ Not active")
        
        # Sample a point to check metadata
        try:
            points = qdrant_client.scroll(
                collection_name=collection.name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if points:
                sample = points[0]
                print(f"\n  Sample payload keys:")
                for key in sample.payload.keys():
                    print(f"    - {key}")
                
                # Check for metadata
                has_metadata = 'channel_username' in sample.payload or 'video_id' in sample.payload
                if has_metadata:
                    print(f"  📺 Has video metadata: ✅")
                else:
                    print(f"  📺 Has video metadata: ❌")
        except Exception as e:
            print(f"  ⚠️  Could not sample point: {e}")
        
        print()
    
    # Show current configuration
    print(f"{'='*80}")
    print("CURRENT BACKEND CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Collection name: {settings.COLLECTION_NAME}")
    print(f"  Data path: {settings.DATA_PATH}")
    print(f"  Embedding model: {settings.EMBEDDING_MODEL}")
    print(f"  Top K: {settings.TOP_K}")
    print()
    
    # Recommendations
    print(f"{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    collection_names = [c.name for c in collections]
    
    if "instructions_v2" in collection_names:
        if settings.COLLECTION_NAME != "instructions_v2":
            print("  ⚠️  You have 'instructions_v2' but are using 'instructions'")
            print("  💡 To use the unified database with metadata:")
            print("     1. Add to .env: COLLECTION_NAME=instructions_v2")
            print("     2. Add to .env: DATA_PATH=data/data_merged.json")
            print("     3. Restart backend")
        else:
            print("  ✅ You're using the unified database with metadata!")
    else:
        print("  ℹ️  'instructions_v2' not found")
        print("  💡 To create unified database:")
        print("     Run: python scripts/merge_and_create_embeddings.py")
    
    print()


if __name__ == "__main__":
    check_collections()
