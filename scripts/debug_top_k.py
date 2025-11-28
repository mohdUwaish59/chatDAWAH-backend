"""
Debug script to check TOP_K configuration
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings


def debug_top_k():
    """Check TOP_K configuration"""
    
    print("="*80)
    print("TOP_K CONFIGURATION DEBUG")
    print("="*80)
    
    print(f"\n📊 Current Configuration:")
    print(f"  TOP_K from settings: {settings.TOP_K}")
    print(f"  Type: {type(settings.TOP_K)}")
    
    # Check environment variable directly
    env_top_k = os.getenv('TOP_K')
    print(f"\n🔍 Environment Variable:")
    print(f"  TOP_K from os.getenv: {env_top_k}")
    print(f"  Type: {type(env_top_k)}")
    
    # Check .env file
    print(f"\n📄 .env File Check:")
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if 'TOP_K' in line and not line.strip().startswith('#'):
                    print(f"  Line {i}: {line.strip()}")
                    # Check for leading spaces
                    if line.startswith(' ') or line.startswith('\t'):
                        print(f"  ⚠️  WARNING: Line has leading whitespace!")
    else:
        print(f"  ❌ .env file not found at: {env_path}")
    
    # Check all retrieval settings
    print(f"\n⚙️  All Retrieval Settings:")
    print(f"  TOP_K: {settings.TOP_K}")
    print(f"  SIMILARITY_THRESHOLD: {settings.SIMILARITY_THRESHOLD}")
    print(f"  COLLECTION_NAME: {settings.COLLECTION_NAME}")
    print(f"  DATA_PATH: {settings.DATA_PATH}")
    
    # Test with chatbot service
    print(f"\n🧪 Testing with Chatbot Service:")
    try:
        from app.services.chatbot_qdrant import chatbot_service
        import asyncio
        
        async def test():
            await chatbot_service.initialize()
            
            # Test retrieve_context with default top_k
            print(f"  Calling retrieve_context with default top_k...")
            context = chatbot_service.retrieve_context("test question")
            print(f"  Expected: {settings.TOP_K} results")
            print(f"  Actual: {len(context)} results")
            
            if len(context) == settings.TOP_K:
                print(f"  ✅ TOP_K is working correctly!")
            else:
                print(f"  ⚠️  Mismatch! Check if similarity threshold is filtering results")
            
            # Test with explicit top_k
            print(f"\n  Calling retrieve_context with explicit top_k=10...")
            context = chatbot_service.retrieve_context("test question", top_k=10)
            print(f"  Expected: 10 results")
            print(f"  Actual: {len(context)} results")
            
            if len(context) == 10:
                print(f"  ✅ Explicit top_k is working!")
            else:
                print(f"  ⚠️  Got {len(context)} results (may be filtered by similarity threshold)")
        
        asyncio.run(test())
        
    except Exception as e:
        print(f"  ❌ Error testing chatbot service: {e}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if settings.TOP_K == 10:
        print("  ✅ TOP_K is set to 10 in configuration")
        print("  ✅ Backend will use 10 as default")
        print("\n  Next steps:")
        print("  1. Restart backend: python main.py")
        print("  2. Test query and check logs")
        print("  3. Verify frontend is sending top_k=10")
    else:
        print(f"  ⚠️  TOP_K is {settings.TOP_K}, not 10")
        print("\n  To fix:")
        print("  1. Check .env file has: TOP_K=10 (no leading spaces)")
        print("  2. Restart backend")
        print("  3. Run this script again")
    
    print()


if __name__ == "__main__":
    debug_top_k()
