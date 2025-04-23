
"""
Implementation of ToG-2.0 (Tree of Goals) agent for enhanced graph traversal in HippoRAG2.
This module replaces the PPR (Personalized PageRank) approach with ToG-2.0's goal-oriented graph exploration.

Reference:
- ToG-2.0 paper: https://openreview.net/pdf?id=oFBu7qaZpS
- ToG-2.0 code: https://github.com/IDEA-FinAI/ToG-2
"""

import os
import json
import time
import networkx as nx
from typing import List, Dict, Tuple, Set, Any, Optional, Union
from collections import deque

class ToGAgent:
    """
    ToG-2.0 agent implementation.
    
    ToG-2.0 is a goal-oriented agent that decomposes complex problems into sub-goals,
    constructs a goal tree, and traverses that tree to solve the problem.
    """
    
    def __init__(self, 
                 llm_client,
                 max_iterations: int = 10,
                 max_children: int = 3,
                 traversal_strategy: str = "dfs",
                 verbose: bool = False):
        """
        Initialize the ToG-2.0 agent.
        
        Args:
            llm_client: LLM API client (used for goal decomposition and synthesis)
            max_iterations: Maximum number of iterations for the agent
            max_children: Maximum number of sub-goals to generate for each goal
            traversal_strategy: Tree traversal strategy ("dfs", "bfs", or "priority")
            verbose: Whether to log detailed information
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.max_children = max_children
        self.traversal_strategy = traversal_strategy
        self.verbose = verbose
        
        # Initialize goal tree
        self.goal_tree = nx.DiGraph()
        self.root_goal_id = None
        self.current_goal_id = 0
        
        # Keep track of processed goals
        self.processed_goals = set()
        
        # Execution trace and retrieved nodes
        self.execution_trace = []
        self.retrieved_nodes = set()
    
    def _create_goal_node(self, goal_description: str, parent_id: Optional[int] = None) -> int:
        """
        Create a goal node and add it to the tree.
        
        Args:
            goal_description: Description of the goal
            parent_id: ID of the parent goal (None for root goals)
            
        Returns:
            ID of the newly created goal
        """
        goal_id = self.current_goal_id
        self.current_goal_id += 1
        
        # Create the node
        self.goal_tree.add_node(
            goal_id, 
            description=goal_description,
            status="pending",
            result=None,
            created_at=time.time()
        )
        
        # Connect to parent if provided
        if parent_id is not None:
            self.goal_tree.add_edge(parent_id, goal_id)
        
        return goal_id
    
    def decompose_goal(self, goal_id: int) -> List[int]:
        """
        Decompose a goal into sub-goals.
        
        Args:
            goal_id: ID of the goal to decompose
            
        Returns:
            List of sub-goal IDs
        """
        goal_description = self.goal_tree.nodes[goal_id]["description"]
        
        # Generate sub-goals using LLM
        subgoals = self._generate_subgoals(goal_description)
        
        # Add each sub-goal to the tree
        subgoal_ids = []
        for subgoal in subgoals[:self.max_children]:
            subgoal_id = self._create_goal_node(subgoal, parent_id=goal_id)
            subgoal_ids.append(subgoal_id)
        
        return subgoal_ids
    
    def _generate_subgoals(self, goal_description: str) -> List[str]:
        """
        Generate sub-goals using LLM.
        
        Args:
            goal_description: Description of the goal
            
        Returns:
            List of sub-goal descriptions
        """
        # Construct prompt for LLM
        prompt = f"""
        I need to solve the following goal:
        {goal_description}
        
        Break down this goal into {self.max_children} smaller, concrete subgoals that would help achieve the main goal.
        List each subgoal on a new line, starting with '- '.
        """
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Extract sub-goals from response
        subgoals = []
        for line in response.strip().split("\n"):
            if line.startswith("- "):
                subgoals.append(line[2:].strip())
        
        return subgoals
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (implementation depends on the LLM being used).
        
        Args:
            prompt: Prompt to send to the LLM
            
        Returns:
            LLM response
        """
        # This method should be replaced with actual LLM client
        try:
            if hasattr(self.llm_client, "generate"):
                # HippoRAG2 LLM client format
                response = self.llm_client.generate(prompt)
                return response.text
            else:
                # Generic LLM client call (e.g., OpenAI)
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call error: {e}")
            # Return simple fallback response in case of error
            return "- Analyze the data\n- Process the results\n- Formulate a conclusion"
    
    def execute_goal(self, goal_id: int, knowledge_graph: nx.Graph) -> Tuple[str, Set[int]]:
        """
        Execute a goal: search the knowledge graph and process the information.
        
        Args:
            goal_id: ID of the goal to execute
            knowledge_graph: Knowledge graph to search
            
        Returns:
            Processing result and set of retrieved node IDs
        """
        goal_description = self.goal_tree.nodes[goal_id]["description"]
        
        # Search knowledge graph for relevant nodes
        relevant_nodes = self._search_knowledge_graph(goal_description, knowledge_graph)
        
        # Extract information from nodes
        retrieved_info = self._extract_info_from_nodes(relevant_nodes, knowledge_graph)
        
        # Process goal with LLM
        result = self._process_goal_with_llm(goal_description, retrieved_info)
        
        # Update goal node
        self.goal_tree.nodes[goal_id]["status"] = "completed"
        self.goal_tree.nodes[goal_id]["result"] = result
        self.goal_tree.nodes[goal_id]["completed_at"] = time.time()
        
        # Return retrieved node IDs
        retrieved_node_ids = {node["id"] for node in relevant_nodes}
        self.retrieved_nodes.update(retrieved_node_ids)
        
        return result, retrieved_node_ids
    
    def _search_knowledge_graph(self, goal_description: str, knowledge_graph: nx.Graph) -> List[Dict]:
        """
        Search knowledge graph for nodes relevant to the goal.
        
        Args:
            goal_description: Description of the goal
            knowledge_graph: Knowledge graph to search
            
        Returns:
            List of relevant nodes
        """
        # Using HippoRAG2's embedding utilities
        from hipporag.knowledge_graph.embedding import get_embedding
        import numpy as np
        
        try:
            # Get goal embedding
            goal_embedding = get_embedding(goal_description)
            
            # Calculate similarity for all nodes
            node_scores = {}
            for node_id in knowledge_graph.nodes():
                node_data = knowledge_graph.nodes[node_id]
                node_text = node_data.get("text", "")
                
                if not node_text:
                    continue
                
                # Get node embedding
                if "embedding" in node_data:
                    node_embedding = node_data["embedding"]
                else:
                    node_embedding = get_embedding(node_text)
                
                # Calculate cosine similarity
                similarity = np.dot(goal_embedding, node_embedding) / (
                    np.linalg.norm(goal_embedding) * np.linalg.norm(node_embedding)
                )
                node_scores[node_id] = float(similarity)
            
            # Sort by score
            sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return top 5 nodes
            relevant_nodes = []
            for node_id, score in sorted_nodes[:5]:
                node_data = knowledge_graph.nodes[node_id]
                relevant_nodes.append({
                    "id": node_id,
                    "text": node_data.get("text", ""),
                    "score": score
                })
            
            return relevant_nodes
        except Exception as e:
            print(f"Error searching knowledge graph: {e}")
            return []
    
    def _extract_info_from_nodes(self, nodes: List[Dict], knowledge_graph: nx.Graph) -> str:
        """
        Extract and format information from nodes.
        
        Args:
            nodes: List of relevant nodes
            knowledge_graph: Knowledge graph
            
        Returns:
            Extracted information string
        """
        extracted_info = ""
        
        for i, node in enumerate(nodes):
            node_id = node["id"]
            node_text = node["text"]
            
            # Add node information
            extracted_info += f"[Document {i+1}]: {node_text}\n\n"
            
            # Add information from neighboring nodes (optional)
            neighbors = list(knowledge_graph.neighbors(node_id))
            if neighbors:
                extracted_info += f"Related information:\n"
                for j, neighbor_id in enumerate(neighbors[:3]):  # Max 3 neighbors
                    neighbor_text = knowledge_graph.nodes[neighbor_id].get("text", "")
                    extracted_info += f"  - {neighbor_text[:100]}...\n"
                extracted_info += "\n"
        
        return extracted_info
    
    def _process_goal_with_llm(self, goal_description: str, retrieved_info: str) -> str:
        """
        Process goal with LLM.
        
        Args:
            goal_description: Description of the goal
            retrieved_info: Retrieved information
            
        Returns:
            Processing result
        """
        prompt = f"""
        Goal: {goal_description}
        
        Available information:
        {retrieved_info}
        
        Based on the information provided, please help me achieve this goal.
        Provide a concise answer that directly addresses the goal.
        """
        
        # Call LLM
        response = self._call_llm(prompt)
        return response.strip()
    
    def solve(self, initial_goal: str, knowledge_graph: nx.Graph) -> Dict[str, Any]:
        """
        Execute ToG process starting from initial goal.
        
        Args:
            initial_goal: Initial goal description
            knowledge_graph: Knowledge graph to search
            
        Returns:
            Dict with result and metadata
        """
        # Create initial goal
        self.root_goal_id = self._create_goal_node(initial_goal)
        
        # Initialize goal queue (depends on strategy)
        if self.traversal_strategy == "dfs":
            goal_queue = [self.root_goal_id]  # Use as stack
        elif self.traversal_strategy == "bfs":
            goal_queue = deque([self.root_goal_id])  # Use as queue
        else:  # priority
            goal_queue = [(0, self.root_goal_id)]  # Priority queue: (depth, goal_id)
        
        iteration = 0
        
        # Main loop
        while goal_queue and iteration < self.max_iterations:
            iteration += 1
            
            # Select next goal based on strategy
            if self.traversal_strategy == "dfs":
                current_goal_id = goal_queue.pop()  # Pop from stack
            elif self.traversal_strategy == "bfs":
                current_goal_id = goal_queue.popleft()  # Pop from queue
            else:  # priority
                _, current_goal_id = goal_queue.pop(0)  # Pop from priority queue
            
            if current_goal_id in self.processed_goals:
                continue
            
            # Check if goal is a leaf node
            is_leaf = self.goal_tree.out_degree(current_goal_id) == 0
            
            if is_leaf:
                # For leaf nodes, either decompose or execute
                if current_goal_id != self.root_goal_id and \
                   len(list(self.goal_tree.predecessors(current_goal_id))) > 0:
                    # Execute non-root leaf nodes with parents
                    result, retrieved_nodes = self.execute_goal(current_goal_id, knowledge_graph)
                    
                    # Add to execution trace
                    self.execution_trace.append({
                        "goal_id": current_goal_id,
                        "description": self.goal_tree.nodes[current_goal_id]["description"],
                        "result": result,
                        "retrieved_nodes": list(retrieved_nodes)
                    })
                    
                    self.processed_goals.add(current_goal_id)
                else:
                    # Decompose root or parentless goals
                    subgoal_ids = self.decompose_goal(current_goal_id)
                    
                    if not subgoal_ids:  # Failed to generate sub-goals
                        # Just execute
                        result, retrieved_nodes = self.execute_goal(current_goal_id, knowledge_graph)
                        
                        # Add to execution trace
                        self.execution_trace.append({
                            "goal_id": current_goal_id,
                            "description": self.goal_tree.nodes[current_goal_id]["description"],
                            "result": result,
                            "retrieved_nodes": list(retrieved_nodes)
                        })
                        
                        self.processed_goals.add(current_goal_id)
                    else:
                        # Add sub-goals to queue
                        if self.traversal_strategy == "dfs":
                            # DFS: Add in reverse (last added processed first)
                            for subgoal_id in reversed(subgoal_ids):
                                goal_queue.append(subgoal_id)
                        elif self.traversal_strategy == "bfs":
                            # BFS: Add in order
                            for subgoal_id in subgoal_ids:
                                goal_queue.append(subgoal_id)
                        else:  # priority
                            # Priority: Add with current depth + 1
                            current_depth = next((d for d, gid in goal_queue if gid == current_goal_id), 0)
                            for subgoal_id in subgoal_ids:
                                goal_queue.append((current_depth+1, subgoal_id))
            else:
                # For non-leaf nodes, check if all children are processed
                all_children_processed = True
                for child_id in self.goal_tree.successors(current_goal_id):
                    if child_id not in self.processed_goals:
                        all_children_processed = False
                        break
                
                if all_children_processed:
                    # Synthesize results from children
                    child_results = []
                    for child_id in self.goal_tree.successors(current_goal_id):
                        child_results.append(self.goal_tree.nodes[child_id]["result"])
                    
                    # Process goal with synthesized information
                    combined_info = "\n".join([f"Subgoal result {i+1}: {result}" 
                                             for i, result in enumerate(child_results)])
                    
                    result = self._process_goal_with_llm(
                        self.goal_tree.nodes[current_goal_id]["description"], 
                        combined_info
                    )
                    
                    # Update goal
                    self.goal_tree.nodes[current_goal_id]["status"] = "completed"
                    self.goal_tree.nodes[current_goal_id]["result"] = result
                    self.goal_tree.nodes[current_goal_id]["completed_at"] = time.time()
                    
                    # Add to execution trace
                    self.execution_trace.append({
                        "goal_id": current_goal_id,
                        "description": self.goal_tree.nodes[current_goal_id]["description"],
                        "result": result,
                        "combined_from_subgoals": True
                    })
                    
                    self.processed_goals.add(current_goal_id)
                else:
                    # Re-add to queue if not all children processed
                    if self.traversal_strategy == "dfs":
                        goal_queue.append(current_goal_id)
                    elif self.traversal_strategy == "bfs":
                        goal_queue.append(current_goal_id)
                    else:  # priority
                        # Increase depth in priority queue
                        current_depth = next((d for d, gid in goal_queue if gid == current_goal_id), 0)
                        goal_queue.append((current_depth+1, current_goal_id))
                    continue
            
            # Re-add parent goals to queue
            if current_goal_id != self.root_goal_id:
                parents = list(self.goal_tree.predecessors(current_goal_id))
                if parents:
                    parent_id = parents[0]
                    if self.traversal_strategy == "dfs":
                        if parent_id not in goal_queue:
                            goal_queue.append(parent_id)
                    elif self.traversal_strategy == "bfs":
                        if parent_id not in goal_queue:
                            goal_queue.append(parent_id)
                    else:  # priority
                        # Parents have lower depth
                        parent_already_in_queue = False
                        for i, (d, gid) in enumerate(goal_queue):
                            if gid == parent_id:
                                parent_already_in_queue = True
                                break
                        if not parent_already_in_queue:
                            current_depth = next((d for d, gid in goal_queue if gid == current_goal_id), 0)
                            goal_queue.append((max(0, current_depth-1), parent_id))
        
        # Return final result and metadata
        final_result = self.goal_tree.nodes[self.root_goal_id].get("result", "No result")
        
        return {
            "result": final_result,
            "retrieved_nodes": list(self.retrieved_nodes),
            "execution_trace": self.execution_trace,
            "goal_tree": nx.node_link_data(self.goal_tree)
        }