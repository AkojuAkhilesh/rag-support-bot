from src.core.vectordb import similarity_search

if __name__ == "__main__":
    query = "What is the refund window?"
    hits = similarity_search(query, k=3)

    print(f"\n🔎 QUERY: {query}\n")
    if not hits:
        print("No results found — check if ingestion ran successfully.")
    else:
        for i, h in enumerate(hits, 1):
            print(f"[{i}] score={h['score']:.3f} source={h['meta'].get('source')}")
            print(h['text'][:200].replace('\n', ' '), "\n")
