"""
Merge data.json and data_new.json, then create unified vector embeddings in Qdrant
Handles different formats:
- data.json: {instruction, input, output}
- data_new.json: {instruction, input, output, channel_username, video_id}
"""
import json
import asyncio
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed.embedding import DefaultEmbedding
from app.core.config import settings


def load_and_normalize_data(file_path: str, has_metadata: bool = False) -> List[Dict[str, Any]]:
    """Load JSON data and normalize format"""
    print(f"Loading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    normalized_data = []
    for item in data:
        normalized_item = {
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
            'output': item.get('output', ''),
            'channel_username': item.get('channel_username', None),
            'video_id': item.get('video_id', None),
            'source': 'data_new.json' if has_metadata else 'data.json'
        }
        normalized_data.append(normalized_item)
    
    print(f"  Loaded {len(normalized_data)} items")
    return normalized_data


def merge_datasets(old_data: List[Dict], new_data: List[Dict]) -> List[Dict]:
    """Merge two datasets and remove duplicates"""
    print("\nMerging datasets...")
    
    # Create a set of unique instructions to detect duplicates
    seen_instructions = set()
    merged_data = []
    
    # Add old data first
    for item in old_data:
        instruction = item['instruction'].strip().lower()
        if instruction and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            merged_data.append(item)
    
    # Add new data, skipping duplicates
    duplicates = 0
    for item in new_data:
        instruction = item['instruction'].strip().lower()
        if instruction and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            merged_data.append(item)
        else:
            duplicates += 1
    
    print(f"  Old data: {len(old_data)} items")
    print(f"  New data: {len(new_data)} items")
    print(f"  Duplicates removed: {duplicates}")
    print(f"  Total merged: {len(merged_data)} items")
    
    return merged_data


def save_merged_data(data: List[Dict], output_path: str):
    """Save merged data to file"""
    print(f"\nSaving merged data to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved successfully!")


async def create_qdrant_collection(
    collection_name: str,
    data: List[Dict],
    qdrant_client: QdrantClient,
    embedding_model: DefaultEmbedding
):
    """Create a new Qdrant collection with embeddings"""
    
    print(f"\n{'='*80}")
    print(f"Creating Qdrant Collection: {collection_name}")
    print(f"{'='*80}")
    
    # Get vector dimension
    print("Getting vector dimensions...")
    sample_embedding = list(embedding_model.embed(["test"]))[0]
    vector_size = len(sample_embedding)
    print(f"  Vector dimension: {vector_size}")
    
    # Check if collection exists
    collections = qdrant_client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists:
        print(f"\n⚠️  Collection '{collection_name}' already exists!")
        response = input("Do you want to delete and recreate it? (yes/no): ")
        if response.lower() == 'yes':
            print(f"Deleting collection '{collection_name}'...")
            qdrant_client.delete_collection(collection_name)
            print("  Deleted!")
        else:
            print("Aborting...")
            return
    
    # Create collection
    print(f"\nCreating collection '{collection_name}'...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print("  Collection created!")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    instructions = [item['instruction'] for item in data]
    embeddings = list(embedding_model.embed(instructions))
    print(f"  Generated {len(embeddings)} embeddings")
    
    # Create points with metadata
    print("\nCreating points with metadata...")
    points = []
    for idx, (item, embedding) in enumerate(zip(data, embeddings)):
        # Build payload with all available metadata
        payload = {
            'instruction': item['instruction'],
            'output': item['output'],
            'source': item['source']
        }
        
        # Add optional metadata if available
        if item.get('channel_username'):
            payload['channel_username'] = item['channel_username']
        if item.get('video_id'):
            payload['video_id'] = item['video_id']
        if item.get('input'):
            payload['input'] = item['input']
        
        points.append(
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=payload
            )
        )
    
    print(f"  Created {len(points)} points")
    
    # Upload in batches
    print("\nUploading to Qdrant...")
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"  Uploaded {min(i + batch_size, len(points))}/{len(points)} items")
    
    print("\n✅ Collection created successfully!")
    
    # Verify
    collection_info = qdrant_client.get_collection(collection_name)
    print(f"\nCollection Info:")
    print(f"  Name: {collection_name}")
    print(f"  Points: {collection_info.points_count}")
    print(f"  Vector size: {collection_info.config.params.vectors.size}")
    print(f"  Distance: {collection_info.config.params.vectors.distance}")


async def test_search(
    collection_name: str,
    qdrant_client: QdrantClient,
    embedding_model: DefaultEmbedding
):
    """Test the new collection with sample queries"""
    
    print(f"\n{'='*80}")
    print("Testing Search")
    print(f"{'='*80}")
    
    test_queries = [
        "What is Islam?",
        "Is the Quran a book of science?",
        "What is the burden of proof?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        
        # Generate query embedding
        query_embedding = list(embedding_model.embed([query]))[0]
        
        # Search using query_points (official API)
        search_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=3
        )
        
        for i, result in enumerate(search_response.points, 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"   Instruction: {result.payload['instruction'][:100]}...")
            print(f"   Source: {result.payload.get('source', 'unknown')}")
            if result.payload.get('channel_username'):
                print(f"   Channel: {result.payload['channel_username']}")
            if result.payload.get('video_id'):
                print(f"   Video ID: {result.payload['video_id']}")


async def main():
    """Main execution"""
    print("="*80)
    print("UNIFIED VECTOR EMBEDDINGS CREATOR")
    print("="*80)
    
    # Configuration
    OLD_DATA_PATH = "data/data.json"
    NEW_DATA_PATH = "data/data_new.json"
    MERGED_DATA_PATH = "data/data_merged.json"
    NEW_COLLECTION_NAME = "instructions_v2"  # New collection name
    
    print(f"\nConfiguration:")
    print(f"  Old data: {OLD_DATA_PATH}")
    print(f"  New data: {NEW_DATA_PATH}")
    print(f"  Merged output: {MERGED_DATA_PATH}")
    print(f"  Collection name: {NEW_COLLECTION_NAME}")
    print(f"  Embedding model: {settings.EMBEDDING_MODEL}")
    
    # Step 1: Load data
    print(f"\n{'='*80}")
    print("STEP 1: Loading Data")
    print(f"{'='*80}")
    
    old_data = load_and_normalize_data(OLD_DATA_PATH, has_metadata=False)
    new_data = load_and_normalize_data(NEW_DATA_PATH, has_metadata=True)
    
    # Step 2: Merge datasets
    print(f"\n{'='*80}")
    print("STEP 2: Merging Datasets")
    print(f"{'='*80}")
    
    merged_data = merge_datasets(old_data, new_data)
    
    # Step 3: Save merged data
    print(f"\n{'='*80}")
    print("STEP 3: Saving Merged Data")
    print(f"{'='*80}")
    
    save_merged_data(merged_data, MERGED_DATA_PATH)
    
    # Step 4: Initialize Qdrant and embedding model
    print(f"\n{'='*80}")
    print("STEP 4: Initializing Services")
    print(f"{'='*80}")
    
    if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
        print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        return
    
    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    print("  Connected!")
    
    print(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
    embedding_model = DefaultEmbedding(model_name=settings.EMBEDDING_MODEL)
    print("  Loaded!")
    
    # Step 5: Create Qdrant collection
    print(f"\n{'='*80}")
    print("STEP 5: Creating Qdrant Collection")
    print(f"{'='*80}")
    
    await create_qdrant_collection(
        collection_name=NEW_COLLECTION_NAME,
        data=merged_data,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model
    )
    
    # Step 6: Test search
    print(f"\n{'='*80}")
    print("STEP 6: Testing Search")
    print(f"{'='*80}")
    
    await test_search(NEW_COLLECTION_NAME, qdrant_client, embedding_model)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully created unified vector database!")
    print(f"\nDetails:")
    print(f"  Collection name: {NEW_COLLECTION_NAME}")
    print(f"  Total items: {len(merged_data)}")
    print(f"  Old data items: {len(old_data)}")
    print(f"  New data items: {len(new_data)}")
    print(f"  Merged data saved to: {MERGED_DATA_PATH}")
    print(f"\nMetadata preserved:")
    print(f"  ✅ instruction, input, output (all items)")
    print(f"  ✅ channel_username, video_id (new items only)")
    print(f"  ✅ source (data.json or data_new.json)")
    print(f"\nNext steps:")
    print(f"  1. Update COLLECTION_NAME in .env to '{NEW_COLLECTION_NAME}'")
    print(f"  2. Update DATA_PATH in .env to '{MERGED_DATA_PATH}' (optional)")
    print(f"  3. Restart your backend server")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
