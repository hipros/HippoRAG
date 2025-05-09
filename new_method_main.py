# experiments/hipporag2_pipeline.py
"""
Unified experiment and evaluation pipeline for HippoRAG2 baselines and five improved methods.

Supported Methods:
  - baseline: Original HippoRAG2 PPR search
  - montecarlo: Monte Carlo Approximate PPR
  - bidirectional: Bidirectional PPR (Lofgren et al.)
  - iterative: ToG-2 style iterative document↔graph retrieval
  - dynamic: Dynamic teleportation Personalized PageRank
  - rp_ep: Relation Prune + Entity Prune (ToG-2 PPR replacement)

Supported Datasets:
  - hotpotqa
  - musique
  - 2wikimultihopqa

Usage Example:
```bash
python experiments/hipporag2_pipeline.py \
  --methods baseline montecarlo bidirectional iterative dynamic rp_ep \
  --datasets hotpotqa musique 2wikimultihopqa \
  --num_walks 1000 --walk_length 10 --push_eps 1e-4 \
  --hops 3 --iters 20 --top_m 5
```
This script computes both retrieval (Recall@2, Recall@5) and QA (EM, F1) metrics and prints LaTeX tables.
All logging.info messages are sent to stderr for separate log capture.
"""

import argparse
import random
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm

# Configure root logger: INFO-level messages to stderr
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Core components from HippoRAG2
from hipporag2 import KnowledgeGraph, Retriever, LLMReader
from hipporag2.evaluation.retrieval_eval import compute_recall
from hipporag2.evaluation.qa_eval import compute_em_f1
from hipporag2.datasets import load_hotpotqa, load_musique, load_2wikimultihopqa
# Baseline and RP+EP override class
from hipporag2.hipporag import HippoRAG as OrigHippoRAG

# --------------------- Method Implementations ---------------------

def monte_carlo_ppr(graph, seed, alpha, num_walks, walk_length):
    """
    Monte Carlo Approximate Personalized PageRank:
      1. Performs `num_walks` random walks from the seed node.
      2. Each walk continues up to `walk_length` steps, teleporting back to seed with prob (1-alpha).
      3. Counts node visits to approximate PPR distribution.
    """
    logging.info(f"[MonteCarloPPR] seed={seed}, walks={num_walks}, length={walk_length}")
    # If seed is not in the graph, return zero vector
    if seed not in graph.nodes():
        logging.warning(f"Seed {seed} not in graph; returning zero vector.")
        return {n: 0.0 for n in graph.nodes()}
    # Initialize visit counts
    counts = {n: 0 for n in graph.nodes()}
    for _ in range(num_walks):
        current = seed
        counts[current] += 1
        for _ in range(walk_length):
            nbrs = list(graph.successors(current))
            # Move to random neighbor with probability alpha
            if random.random() < alpha and nbrs:
                current = random.choice(nbrs)
            else:
                # Teleport back to seed
                current = seed
            counts[current] += 1
    # Normalize counts to get probabilities
    total = sum(counts.values())
    return {n: counts[n] / total for n in counts}


def bidirectional_ppr(graph, source, target, alpha, push_eps, num_walks):
    """
    Bidirectional PPR (Lofgren et al.):
      1. Reverse local-push from target to build residuals p[v].
      2. Forward Monte Carlo from source, accumulating p[u] on teleport.
      3. Returns approximate PPR(source->target).
    """
    logging.info(f"[BidirectionalPPR] source={source}, target={target}, walks={num_walks}")
    nodes = list(graph.nodes())
    # Initialize residual and p-vectors
    residual = {n: 0.0 for n in nodes}
    p = {n: 0.0 for n in nodes}
    # If target missing, return zero
    if target not in residual:
        logging.warning(f"Target {target} not in graph; returning zero.")
        return 0.0
    residual[target] = 1.0
    queue = [target]
    # Reverse local-push phase
    while queue:
        v = queue.pop(0)
        rv = residual[v]
        p[v] += alpha * rv
        for u in graph.predecessors(v):
            deg = graph.out_degree(u)
            # Skip zero-degree nodes to avoid division by zero
            if deg == 0:
                continue
            inc = (1 - alpha) * rv / deg
            if inc > push_eps:
                residual[u] += inc
                queue.append(u)
        residual[v] = 0
    # Forward random-walk phase
    est = 0.0
    for _ in range(num_walks):
        u = source
        while True:
            nbrs = list(graph.successors(u))
            if random.random() < alpha and nbrs:
                u = random.choice(nbrs)
                if u == target:
                    est += 1
                    break
            else:
                est += residual.get(u, 0.0)
                break
    return est / num_walks


def iterative_rag(query, hops, retriever, kg, reader):
    """
    Iterative RAG (ToG-2 style):
      1. Retrieve initial documents for query.
      2. Extract entities and fetch their KG neighbors.
      3. Re-query merged neighbor set for `hops` iterations.
    """
    logging.info(f"[IterativeRAG] query='{query}', hops={hops}")
    docs = retriever.search(query)
    for _ in range(hops):
        entities = reader.extract_entities(" ".join(docs))
        neighbors = []
        for e in entities:
            # Defend against None
            neighbors.extend(kg.neighbors(e) or [])
        docs = retriever.search(" ".join(neighbors))
    return docs


def dynamic_teleport_ppr(graph, query, reader, alpha, iters, top_m):
    """
    Dynamic Teleportation PPR:
      1. Obtain query embedding via `reader.embed` or fallback to `reader.encode`.
      2. Compute cosine similarity with node embeddings to select top_m seeds.
      3. Build teleport vector and run power iteration for `iters`.
    """
    logging.info(f"[DynamicTeleportPPR] query='{query}', iters={iters}, top_m={top_m}")
    # Obtain query embedding with fallback
    try:
        q_emb = reader.embed(query)
    except AttributeError:
        q_emb = reader.encode(query)
    # Ensure node embeddings exist
    node_embs = getattr(graph, 'node_embeddings', {})
    if not node_embs:
        logging.warning("No node embeddings; returning zero vector.")
        return {n: 0.0 for n in graph.nodes()}
    # Compute similarity scores
    scores = {
        n: np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-10)
        for n, emb in node_embs.items()
    }
    # Select top_m seeds
    seeds = sorted(scores, key=scores.get, reverse=True)[:top_m]
    teleport = {n: (scores[n] if n in seeds else 0.0) for n in graph.nodes()}
    s = sum(teleport.values()) or 1.0
    teleport = {n: teleport[n] / s for n in teleport}
    # Power iteration
    p = teleport.copy()
    for _ in range(iters):
        new_p = {n: (1 - alpha) * teleport[n] for n in teleport}
        for u in graph.nodes():
            deg = graph.out_degree(u)
            if deg == 0:
                continue
            for v in graph.successors(u):
                new_p[v] += alpha * p[u] / deg
        p = new_p
    return p

# --------------------- RP+EP Implementation ---------------------

def graph_search_rpep(self, query, query_embedding):
    """
    Relation Prune + Entity Prune search:
      1. Seed from top_k_facts or entity embeddings.
      2. For max_depth=2:
         - Score relations via LLM prompt.
         - Prune entities by chunk embedding similarity.
    """
    logging.info(f"[RP+EP] query='{query}'")
    current = set()
    # Initialize seeds
    if getattr(self, 'top_k_facts', None):
        for f in self.top_k_facts:
            for ent in (f['subject'], f['object']):
                if ent in self.graph.vs['name']:
                    current.add(ent)
    else:
        seeds = self.entity_embedding_store.search(query_embedding, k=self.global_config.linking_top_k)
        current.update(seeds)
    pruned = []
    # RP+EP loop
    for _ in range(2):
        sel_rel = {}
        next_ent = set()
        # Relation prune via LLM
        for ent in current:
            outs = self.graph.incident(ent, mode='out')
            rels = {self.graph.es[e]['relation'] for e in outs if self.graph.es[e]['relation'].lower() != 'synonym'}
            if not rels:
                continue
            prompt = f"Entity: {self.graph.vs[ent]['name']}\nQuestion: {query}\nRelations:\n"
            prompt += '\n'.join(f"- {r}" for r in rels)
            resp = self.llm_model.generate(prompt + "\nRate 0-10 and list top ones.")
            scores_map = {}
            # Parse LLM response
            for line in resp.splitlines():
                if ':' in line:
                    r, s = line.split(':', 1)
                    try:
                        scores_map[r.strip()] = float(s.strip())
                    except:
                        pass
            top_rels = [r for r, sc in sorted(scores_map.items(), key=lambda x: -x[1]) if sc >= 2][:5]
            # Collect next-hop entities
            for e in outs:
                if self.graph.es[e]['relation'] in top_rels:
                    next_ent.add(self.graph.es[e].target)
        # Entity prune via chunk embeddings
        ent_scores = []
        for ent in next_ent:
            name = self.graph.vs[ent]['name']
            results = self.chunk_embedding_store.search(f"{name} {query}", top_k=10)
            score = results[0].score if results else 0
            ent_scores.append((ent, score, results))
        # Keep top-10 entities
        ent_scores.sort(key=lambda x: x[1], reverse=True)
        pruned = ent_scores[:10]
        current = {e for e, _, _ in pruned}
        if not current:
            break
    # Aggregate final passages
    final = [(res[0].id, res[0].score) for _, _, res in pruned if res]
    final.sort(key=lambda x: -x[1])
    ids, scores = zip(*final) if final else ((), ())
    return list(ids), list(scores)

# Monkey-patch the method onto OrigHippoRAG
OrigHippoRAG.graph_search_rpep = graph_search_rpep

# --------------------- Evaluation Functions ---------------------

def evaluate_retrieval(method, graph, retriever, reader, queries, gt_passages, args, hippo_baseline):
    """
    Runs retrieval for each query and computes Recall@2 and Recall@5.
    """
    logging.info(f"[EvalRetrieval] method={method}")
    recalls = {2: [], 5: []}
    # Progress bar for interactive monitoring
    for q, gt in tqdm(list(zip(queries, gt_passages)), desc=f"Retrieval({method})"):
        if method == 'baseline':
            ids, _ = hippo_baseline.graph_search(q)
            preds = ids
        elif method == 'montecarlo':
            scores = monte_carlo_ppr(graph, q, args.alpha, args.num_walks, args.walk_length)
            preds = retriever.rank_by_ppr(scores)
        elif method == 'bidirectional':
            scores = {t: bidirectional_ppr(graph, q, t, args.alpha, args.push_eps, args.num_walks) for t in graph.nodes()}
            preds = retriever.rank_by_ppr(scores)
        elif method == 'iterative':
            preds = iterative_rag(q, args.hops, retriever, graph, reader)
        elif method == 'dynamic':
            scores = dynamic_teleport_ppr(graph, q, reader, args.alpha, args.iters, args.top_m)
            preds = retriever.rank_by_ppr(scores)
        else:
            emb = hippo_baseline.embed_query(q)
            preds, _ = hippo_baseline.graph_search_rpep(q, emb)
        rec = compute_recall(preds, gt, ks=[2,5])
        for k in recalls:
            recalls[k].append(rec[k])
    return {k: np.mean(v) for k, v in recalls.items()}


def evaluate_qa(method, retriever, reader, queries, gt_answers, hippo_baseline):
    """
    Runs QA for each query and computes EM and F1 metrics.
    """
    logging.info(f"[EvalQA] method={method}")
    ems, f1s = [], []
    for q, refs in zip(queries, gt_answers):
        if method == 'baseline':
            preds, _ = hippo_baseline.graph_search(q)
        elif method in ['montecarlo', 'bidirectional', 'dynamic']:
            # Compute PPR scores and rank passages
            if method == 'montecarlo':
                scores = monte_carlo_ppr(graph, q, args.alpha, args.num_walks, args.walk_length)
            elif method == 'bidirectional':
                scores = {t: bidirectional_ppr(graph, q, t, args.alpha, args.push_eps, args.num_walks) for t in graph.nodes()}
            else:  # dynamic
                scores = dynamic_teleport_ppr(graph, q, reader, args.alpha, args.iters, args.top_m)
            preds = retriever.rank_by_ppr(scores)
        elif method == 'iterative':
            preds = retriever.search(q)
        else:  # rp_ep
            preds, _ = hippo_baseline.graph_search_rpep(q, hippo_baseline.embed_query(q))
        ans = reader.predict(q, preds)
        em, f1 = compute_em_f1([ans], [refs])
        ems.append(em)
        f1s.append(f1)
    return float(np.mean(ems)), float(np.mean(f1s))

# --------------------- Main Pipeline ---------------------

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', required=True,
                        choices=['baseline', 'montecarlo', 'bidirectional', 'iterative', 'dynamic', 'rp_ep'])
    parser.add_argument('--datasets', nargs='+', required=True,
                        choices=['hotpotqa', 'musique', '2wikimultihopqa'])
    parser.add_argument('--num_walks', type=int, default=1000)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.85)
    parser.add_argument('--push_eps', type=float, default=1e-4)
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--top_m', type=int, default=5)
    args = parser.parse_args()

    logging.info("[Main] Loading resources")
    # Load Knowledge Graph and convert to NetworkX
    kg = KnowledgeGraph.load_from_dir('data/kg/')
    graph = kg.to_networkx()
    graph.node_embeddings = kg.node_embeddings  # Attach precomputed embeddings
    # Initialize retriever and LLM reader
    retriever = Retriever.from_hipporag2('data/index/')
    reader = LLMReader.from_pretrained('gpt-4o-mini')
    # Instantiate baseline HippoRAG model
    hippo_baseline = OrigHippoRAG.from_config('config.yaml')

    # Map dataset names to loader functions
    loader_map = {
        'hotpotqa': load_hotpotqa,
        'musique': load_musique,
        '2wikimultihopqa': load_2wikimultihopqa
    }
    results = {}
    # Iterate over datasets and methods
    for ds in args.datasets:
        logging.info(f"[Main] Dataset: {ds}")
        qs, gt_ps, gt_ans = loader_map[ds](split='test')  # Load queries, passage GTs, answer GTs
        results[ds] = {}
        for m in args.methods:
            logging.info(f"[Main] Method: {m}")
            # Evaluate retrieval metrics
            r = evaluate_retrieval(m, graph, retriever, reader, qs, gt_ps, args, hippo_baseline)
            # Evaluate QA metrics
            em, f1 = evaluate_qa(m, retriever, reader, qs, gt_ans, hippo_baseline)
            results[ds][m] = {'R@2': r[2], 'R@5': r[5], 'EM': em, 'F1': f1}
            logging.info(f"[Main] Completed {m} on {ds}")

    # Print LaTeX tables for each dataset
    for ds, res in results.items():
        print(f"% Dataset: {ds}")
        print("\\begin{tabular}{l" + "cccc"*len(args.methods) + "}")
        header = "Method & " + " & ".join([f"{m} R@2 & {m} R@5 & {m} EM & {m} F1" for m in args.methods]) + "\\"
        print(header)
        print("\\midrule")
        for m in args.methods:
            vals = results[ds][m]
            print(f"{m} & {vals['R@2']:.3f} & {vals['R@5']:.3f} & {vals['EM']:.3f} & {vals['F1']:.3f}\\")
        print("\\end{tabular}\n")

if __name__ == '__main__':
    main()
