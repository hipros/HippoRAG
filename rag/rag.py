"""
This is a partial modification to the existing rag.py file to support ToG retriever.
You need to integrate this code with the existing file.
"""

# Add the following in the appropriate location in the HippoRAG class

def __init__(self, config=None, **kwargs):
    """Initialize HippoRAG with either PPR or ToG retriever"""
    # Existing initialization code...
    
    # Update retriever initialization to support ToG
    retriever_type = self.config.get("retriever", {}).get("type", "ppr")
    retriever_config = self.config.get("retriever", {}).get("params", {})
    
    # Use factory to get correct retriever
    from hipporag.knowledge_graph.retriever import get_retriever
    self.retriever = get_retriever(
        self.knowledge_graph, 
        retriever_type=retriever_type,
        config=retriever_config
    )
    
    # Rest of initialization...