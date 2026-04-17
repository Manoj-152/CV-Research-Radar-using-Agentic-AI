import requests
import json

def get_hf_metadata(arxiv_id):
    """
    Queries the Hugging Face Papers API.
    """
    url = f"https://huggingface.co/api/papers/{arxiv_id}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title"),
                "upvotes": data.get("upvotes", 0),
                "comments": len(data.get("comments", [])),
                "is_trending": data.get("upvotes", 0) > 20,
                "hf_link": f"https://huggingface.co/papers/{arxiv_id}"
            }
    except Exception as e:
        print(f"Error fetching HF metadata: {e}")
    
    return None


if __name__ == '__main__':
    test_ids = ["1512.03385", "2403.00504", "2604.09167"]
    
    print("--- Testing Hugging Face Metadata Retrieval ---")
    for arxiv_id in test_ids:
        print(f"\nChecking ArXiv ID: {arxiv_id}...")
        meta = get_hf_metadata(arxiv_id)
        
        if meta:
            print(f"  [+] Title: {meta['title']}")
            print(f"  [+] Upvotes: {meta['upvotes']}")
            print(f"  [+] URL: {meta['hf_link']}")
        else:
            print("  [-] No data found on Hugging Face for this ID.")