
"""
Example usage of ToG-2.0 graph traversal in HippoRAG2.
"""

def example_usage():
    """Demonstrate how to use ToG retriever in HippoRAG2"""
    from hipporag import HippoRAG
    
    # Method 1: Using configuration
    config = {
        "knowledge_graph": {
            "path": "path/to/knowledge_graph"
        },
        "retriever": {
            "type": "tog",
            "params": {
                "max_iterations": 10,
                "max_children": 3,
                "traversal_strategy": "dfs"
            }
        },
        "llm": {
            # LLM configuration
        }
    }
    
    rag = HippoRAG(config=config)
    
    # Method 2: Manual override
    from hipporag.knowledge_graph.tog_retriever import ToGRetriever
    
    rag = HippoRAG()
    tog_retriever = ToGRetriever(
        rag.knowledge_graph,
        llm_client=rag.llm,
        config={
            "max_iterations": 10,
            "max_children": 3,
            "traversal_strategy": "dfs"
        }
    )
    
    # Replace PPR retriever with ToG retriever
    rag.retriever = tog_retriever
    
    # Use in the same way as with PPR retriever
    result = rag.query("What causes climate change?")
    print(result)