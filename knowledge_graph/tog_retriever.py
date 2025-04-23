
"""
ToG-2.0 based retriever for HippoRAG2.
This retriever replaces the PPR (Personalized PageRank) approach with ToG-2.0's goal-oriented graph exploration.
"""

import networkx as nx
from typing import List, Dict, Tuple, Any
from hipporag.knowledge_graph.tog_agent import ToGAgent

class ToGRetriever:
    """
    ToG-2.0 based graph retriever.
    
    Implements the HippoRAG2 retriever interface using ToG-2.0's goal-oriented graph traversal.
    """
    
    def __init__(self, 
                 knowledge_graph: nx.Graph, 
                 llm_client=None,
                 config: Dict[str, Any] = None):
        """
        Initialize ToG retriever.
        
        Args:
            knowledge_graph: Knowledge graph to search
            llm_client: LLM client (if None, will use default from config)
            config: Configuration dictionary
        """
        self.knowledge_graph = knowledge_graph
        self.config = config or {}
        
        # Set LLM client
        self.llm_client = llm_client
        
        # Initialize ToG agent
        self.tog_agent = ToGAgent(
            llm_client=self.llm_client,
            max_iterations=self.config.get("max_iterations", 10),
            max_children=self.config.get("max_children", 3),
            traversal_strategy=self.config.get("traversal_strategy", "dfs"),
            verbose=self.config.get("verbose", False)
        )
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant nodes for query.
        
        Args:
            query: User query
            k: Maximum number of nodes to return
            
        Returns:
            List of retrieved documents
        """
        # Use ToG agent to process query
        result = self.tog_agent.solve(query, self.knowledge_graph)
        
        # Get retrieved node IDs
        retrieved_node_ids = result["retrieved_nodes"]
        
        # Construct document list from node IDs
        retrieved_docs = []
        for node_id in retrieved_node_ids[:k]:  # Limit to k
            if node_id in self.knowledge_graph.nodes:
                node_data = self.knowledge_graph.nodes[node_id]
                doc = {
                    "id": node_id,
                    "text": node_data.get("text", ""),
                    "metadata": node_data.get("metadata", {})
                }
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def retrieve_with_score(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant nodes with scores.
        
        Args:
            query: User query
            k: Maximum number of nodes to return
            
        Returns:
            List of (document, score) tuples
        """
        # Perform standard retrieval
        docs = self.retrieve(query, k)
        
        # Assign scores based on execution trace
        # (ToG doesn't have explicit scores, so use order in execution trace)
        execution_trace = self.tog_agent.execution_trace
        
        # Score nodes based on order of appearance
        node_scores = {}
        for i, entry in enumerate(execution_trace):
            if "retrieved_nodes" in entry:
                for j, node_id in enumerate(entry["retrieved_nodes"]):
                    # Higher score for earlier positions in lists and later stages in execution
                    score = (len(execution_trace) - i) * 0.1 + (len(entry["retrieved_nodes"]) - j) * 0.01
                    node_scores[node_id] = max(score, node_scores.get(node_id, 0))
        
        # Create scored documents
        scored_docs = []
        for doc in docs:
            score = node_scores.get(doc["id"], 0.0)
            scored_docs.append((doc, score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]
