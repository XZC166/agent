import os
import sys
import argparse
from rag_database.embedder import OpenFOAMEmbedder
from agents.blastfoamagent import BlastFoamAgent
import config

def main():
    parser = argparse.ArgumentParser(description="BlastFoam Simulation Agent")
    parser.add_argument("--rebuild-db", action="store_true", help="Rebuild the vector database")
    args = parser.parse_args()

    print("Initializing RAG System...")
    embedder = OpenFOAMEmbedder()
    
    if args.rebuild_db:
        print("Rebuilding vector store...")
        embedder.build_vector_store(force_rebuild=True)
    else:
        # Ensure vector store is loaded
        # Check if vector store file exists, if not build it
        persist_path = config.VECTOR_STORE_DIR / "vector_store.pkl"
        if not persist_path.exists():
             print("Vector store not found. Building from scratch...")
             embedder.build_vector_store()
        else:
             embedder.build_vector_store(force_rebuild=False)

    print("Initializing Agent...")
    agent = BlastFoamAgent()

    print("\n=== BlastFoam Simulation Agent ===")
    print("Type 'exit' or 'quit' to stop.")
    
    # Context cache for multi-turn conversation
    cached_context = None
    chat_history = []

    while True:
        try:
            user_input = input("\nUser Request: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input:
                continue

            # Only retrieve on the first turn
            if cached_context is None:
                print("\nRetrieving relevant cases...")
                relevant_docs = embedder.similarity_search(user_input)
                cached_context = embedder.format_rag_context(relevant_docs)
                
                print(f"Found {len(relevant_docs)} relevant cases.")
                if config.ENABLE_VERBOSE_LOGGING:
                    for doc in relevant_docs:
                        print(f"- {doc.metadata.get('source', 'Unknown')}")
            else:
                print("\nUsing cached context from first turn (skipping retrieval)...")

            print("\nAgent is thinking and executing...")
            
            # Format chat history
            history_str = "\n".join(chat_history)
            
            response = agent.run(user_input, cached_context, history_str)
            
            print("\nAgent Response:")
            print(response)
            
            # Update history
            chat_history.append(f"User: {user_input}")
            chat_history.append(f"Agent: {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
