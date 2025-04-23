
"""
This is a partial modification to the existing retriever.py file to add ToG support.
You need to integrate this code with the existing file.
"""

# Add the following imports at the top of the file
from hipporag.knowledge_graph.tog_retriever import ToGRetriever

# Add the following function to the existing retriever.py file

def get_retriever(knowledge_graph, retriever_type="ppr", config=None):
    """
    Factory function for retrievers.
    
    Args:
        knowledge_graph: Knowledge graph to search
        retriever_type: Type of retriever ("ppr" or "tog")
        config: Configuration dictionary
        
    Returns:
        Retriever instance
    """
    if retriever_type.lower() == "ppr":
        from hipporag.knowledge_graph.ppr_retriever import PPRRetriever
        return PPRRetriever(knowledge_graph, config)
    elif retriever_type.lower() == "tog":
        from hipporag.knowledge_graph.tog_retriever import ToGRetriever
        return ToGRetriever(knowledge_graph, config=config)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
