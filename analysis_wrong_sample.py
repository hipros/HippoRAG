import os
import json
import numpy as np
import pandas as pd
import argparse
import logging
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool, compute_mdhash_id
from src.hipporag.utils.config_utils import BaseConfig
#from src.hipporag.utils.openie_utils import text_processing
from src.hipporag.utils.eval_utils import QAExactMatch, QAF1Score, RetrievalRecall


def deeper_graph_analysis(hipporag, incorrect_data):
    """
    More in-depth analysis of graph exploration
    """
    for sample in incorrect_data:
        query = sample['question']
        
        # 1. 정답 문서와 검색된 문서 사이의 그래프 거리 측정
        if 'gold_docs' in sample and sample['gold_docs']:
            gold_doc_ids = [compute_mdhash_id(doc, prefix="chunk-") for doc in sample['gold_docs']]
            retrieved_doc_ids = [compute_mdhash_id(doc, prefix="chunk-") for doc in sample['retrieved_docs']]
            
            # 그래프에서 최단 경로 찾기
            shortest_paths = []
            for gold_id in gold_doc_ids:
                for ret_id in retrieved_doc_ids[:5]:  # 상위 5개 검색 문서만 확인
                    try:
                        path = hipporag.graph.get_shortest_paths(
                            hipporag.node_name_to_vertex_idx[gold_id], 
                            hipporag.node_name_to_vertex_idx[ret_id], 
                            weights='weight'
                        )
                        if path and len(path[0]) > 0:
                            shortest_paths.append((gold_id, ret_id, len(path[0])-1))
                    except:
                        pass
            
            sample['graph_distance_analysis'] = shortest_paths
        
        # 2. 그래프 커뮤니티 분석
        try:
            # 리트리벌과 관련된 서브그래프 추출
            if 'graph_analysis' in sample and 'top_entities' in sample['graph_analysis']:
                entity_keys = [compute_mdhash_id(entity, prefix="entity-") 
                              for entity in sample['graph_analysis']['top_entities']]
                
                # 서브그래프에서 커뮤니티 탐지 (필요시)
                # communities = hipporag.graph.community_multilevel(weights='weight')
                
                # 엔티티 간 연결 구조 분석
                connections = []
                for i, e1 in enumerate(entity_keys):
                    for e2 in entity_keys[i+1:]:
                        if e1 in hipporag.node_name_to_vertex_idx and e2 in hipporag.node_name_to_vertex_idx:
                            idx1 = hipporag.node_name_to_vertex_idx[e1]
                            idx2 = hipporag.node_name_to_vertex_idx[e2]
                            
                            # 두 엔티티 간 연결 확인
                            if hipporag.graph.are_connected(idx1, idx2):
                                connections.append((e1, e2))
                
                sample['entity_connections'] = connections
        except Exception as e:
            sample['entity_connection_error'] = str(e)
    
    return incorrect_data

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs

def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def analyze_incorrect_samples(hipporag, queries, gold_docs, gold_answers, samples):
    """
    Function to analyze incorrect samples in HippoRAG
    
    Args:
        hipporag: HippoRAG instance
        queries: List of queries
        gold_docs: List of gold documents
        gold_answers: List of gold answers
        samples: Original dataset samples
    
    Returns:
        incorrect_data: List of dictionaries containing detailed information about incorrect samples
    """
    # 1. Execute rag_qa and store results
    logging.info("Executing RAG QA and analyzing results...")
    results = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=gold_answers)
    
    # Extract variables according to result structure
    if len(results) == 5:  # If evaluation results are included
        query_solutions, response_messages, metadata, retrieval_results, qa_results = results
    else:
        query_solutions, response_messages, metadata = results
        retrieval_results = qa_results = None
    
    # 2. Identify incorrect samples
    incorrect_indices = []
    incorrect_data = []
    
    # Initialize F1 scorer
    qa_f1_scorer = QAF1Score()
    
    for i, query_solution in enumerate(query_solutions):
        predicted_answer = query_solution.answer
        gold_answer_list = list(gold_answers[i])
        
        # Use F1 Score to evaluate answer correctness
        # Calculate F1 score for each gold answer and take the maximum
        f1_scores = []
        for gold_ans in gold_answer_list:
            f1 = qa_f1_scorer.calculate(predicted_answer, gold_ans)
            f1_scores.append(f1)
        
        max_f1 = max(f1_scores) if f1_scores else 0
        
        # Consider incorrect if F1 score is below threshold (e.g., 0.5)
        is_correct = max_f1 >= 0.5
        
        if not is_correct:
            incorrect_indices.append(i)
            
            # Collect detailed information about incorrect sample
            incorrect_sample = {
                'index': i,
                'original_sample': samples[i] if i < len(samples) else None,
                'question': query_solution.question,
                'predicted_answer': predicted_answer,
                'gold_answers': gold_answer_list,
                'max_f1_score': max_f1,
                'retrieved_docs': query_solution.docs,
                'doc_scores': query_solution.doc_scores.tolist() if hasattr(query_solution, 'doc_scores') else None,
            }
            
            # Add graph analysis information (rest of your existing code)
            if hipporag.ready_to_retrieve:
                try:
                    query_fact_scores = hipporag.get_fact_scores(query_solution.question)
                    top_k_fact_indices, top_k_facts, rerank_log = hipporag.rerank_facts(query_solution.question, query_fact_scores)
                    
                    incorrect_sample['graph_analysis'] = {
                        'top_k_facts': top_k_facts,
                        'facts_before_rerank': rerank_log.get('facts_before_rerank', [])[:10],
                        'facts_after_rerank': rerank_log.get('facts_after_rerank', [])[:10],
                        'top_entities': extract_top_entities(top_k_facts),
                        'reranking_details': rerank_log
                    }
                    
                    if len(top_k_facts) > 0:
                        try:
                            passage_node_indices = [hipporag.node_name_to_vertex_idx.get(doc_id, -1) 
                                                  for doc_id in hipporag.passage_node_keys]
                            doc_scores = {}
                            for idx, score in zip(passage_node_indices, query_solution.doc_scores):
                                if idx >= 0:
                                    doc_scores[hipporag.graph.vs[idx]['name']] = float(score)
                            
                            incorrect_sample['graph_analysis']['doc_scores_in_graph'] = doc_scores
                        except Exception as e:
                            incorrect_sample['graph_analysis']['ppr_analysis_error'] = str(e)
                except Exception as e:
                    incorrect_sample['graph_analysis_error'] = str(e)
            
            incorrect_data.append(incorrect_sample)
    
    logging.info(f"Out of {len(queries)} samples, {len(incorrect_indices)} were incorrect. ({len(incorrect_indices)/len(queries)*100:.2f}%)")
    
    # 3. Output analysis results for incorrect samples
    if incorrect_data:
        analyze_error_patterns(incorrect_data, gold_docs)
    
    return incorrect_data

def extract_top_entities(top_k_facts):
    """Extract entities from top facts"""
    entities = set()
    for fact in top_k_facts:
        if isinstance(fact, list) and len(fact) >= 3:
            entities.add(fact[0])  # Subject
            entities.add(fact[2])  # Object
    return list(entities)

def analyze_error_patterns(incorrect_data, gold_docs=None):
    """Analysis of error patterns in incorrect samples"""
    logging.info("\n=== Error Pattern Analysis ===")
    
    # 1. Cases where relevant document was retrieved but answer is wrong
    docs_with_answer = 0
    
    for sample in incorrect_data:
        idx = sample['index']
        if gold_docs and idx < len(gold_docs):
            sample_gold_docs = gold_docs[idx]
            retrieved_docs = sample['retrieved_docs']
            
            # Check if any gold document is in retrieved documents
            has_relevant_doc = any(gold_doc in retrieved_docs for gold_doc in sample_gold_docs)
            
            if has_relevant_doc:
                docs_with_answer += 1
    
    if gold_docs:
        logging.info(f"Cases where answer document was retrieved but answer was wrong: {docs_with_answer}/{len(incorrect_data)} ({docs_with_answer/len(incorrect_data)*100:.2f}%)")
        
        # 2. Retrieval errors: cases where answer document was not retrieved
        retrieval_errors = len(incorrect_data) - docs_with_answer
        logging.info(f"Cases where answer document was not retrieved: {retrieval_errors}/{len(incorrect_data)} ({retrieval_errors/len(incorrect_data)*100:.2f}%)")
    
    # 3. Detailed output for top 3 incorrect samples
    logging.info("\n=== Analysis of Top 3 Incorrect Samples ===")
    for i, sample in enumerate(incorrect_data[:3]):
        logging.info(f"\nSample #{i+1}")
        logging.info(f"Question: {sample['question']}")
        logging.info(f"Gold answers: {sample['gold_answers']}")
        logging.info(f"Predicted: {sample['predicted_answer']}")
        logging.info(f"F1 Score: {sample.get('max_f1_score', 'N/A')}")
        logging.info(f"Retrieved documents (top 3):")
        for j, doc in enumerate(sample['retrieved_docs'][:3]):
            logging.info(f"  {j+1}. {doc[:100]}... (score: {sample['doc_scores'][j] if sample['doc_scores'] else 'N/A'})")
        
        if 'graph_analysis' in sample:
            logging.info("Graph analysis:")
            logging.info(f"  Facts before reranking: {len(sample['graph_analysis'].get('facts_before_rerank', []))}")
            logging.info(f"  Facts after reranking: {len(sample['graph_analysis'].get('facts_after_rerank', []))}")
            if sample['graph_analysis'].get('facts_after_rerank'):
                logging.info(f"  Top fact: {sample['graph_analysis']['facts_after_rerank'][0]}")
                if 'top_entities' in sample['graph_analysis']:
                    logging.info(f"  Key entities: {', '.join(sample['graph_analysis']['top_entities'][:5])}")

def save_analysis_results(incorrect_data, filename="incorrect_samples_analysis.json"):
    with open(filename, 'w') as f:
        json.dump(incorrect_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Analysis results saved to {filename}")

def visualize_error_distribution(incorrect_data, output_dir, dataset_name):
    """Visualization of error type distribution"""
    try:
        # Classify error types
        retrieval_errors = 0
        qa_errors = 0
        
        for sample in incorrect_data:
            if 'gold_docs' in sample and sample.get('gold_docs'):
                gold_docs = sample['gold_docs']
                retrieved_docs = sample['retrieved_docs']
                
                has_relevant_doc = any(gold_doc in retrieved_docs for gold_doc in gold_docs)
                
                if has_relevant_doc:
                    qa_errors += 1
                else:
                    retrieval_errors += 1
            else:
                qa_errors += 1
        
        # Visualize with pie chart
        labels = ['Retrieval Error', 'QA Error']
        sizes = [retrieval_errors, qa_errors]
        colors = ['#ff9999','#66b3ff']
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'Error Type Distribution for {dataset_name} Dataset')
        plt.tight_layout()
        
        # Save results
        viz_path = os.path.join(output_dir, f"{dataset_name}_error_distribution.png")
        plt.savefig(viz_path)
        logging.info(f"Error distribution visualization saved to {viz_path}")
        
    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="HippoRAG Error Analysis")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    parser.add_argument('--analysis_dir', type=str, default='error_analysis', help='Error analysis results directory')
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit number of samples to analyze (for testing)')
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    if save_dir == 'outputs':
        save_dir = save_dir + '/' + dataset_name
    else:
        save_dir = save_dir + '_' + dataset_name

    # 분석 결과 저장 디렉토리 생성
    analysis_dir = f"{args.analysis_dir}/{dataset_name}"
    os.makedirs(analysis_dir, exist_ok=True)

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{analysis_dir}/analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    logging.info(f"데이터셋: {dataset_name}, LLM: {llm_name}, 임베딩 모델: {args.embedding_name}")

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    logging.info(f"코퍼스 문서 수: {len(docs)}")

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples_path = f"reproduce/dataset/{dataset_name}.json"
    logging.info(f"데이터셋 로딩: {samples_path}")
    samples = json.load(open(samples_path, "r"))
    
    # 샘플 수 제한 (개발 및 디버깅용)
    if args.limit_samples:
        samples = samples[:args.limit_samples]
        logging.info(f"샘플 수 제한: {args.limit_samples}개")

    all_queries = [s['question'] for s in samples]
    logging.info(f"쿼리 수: {len(all_queries)}")

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
        logging.info(f"골드 문서 및 답변 로드 완료")
    except Exception as e:
        logging.warning(f"골드 문서 로드 중 오류: {str(e)}")
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode
    )

    logging.info("HippoRAG 인스턴스 생성")
    hipporag = HippoRAG(global_config=config)

    # 이미 인덱싱되어 있을 경우 건너뛸 수 있음
    if force_index_from_scratch:
        logging.info("문서 인덱싱 시작")
        hipporag.index(docs)
    else:
        logging.info("기존 인덱스를 사용하여 리트리벌 준비")
        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()

    # 틀린 샘플 분석
    logging.info("틀린 샘플 분석 시작")
    incorrect_data = analyze_incorrect_samples(hipporag, all_queries, gold_docs, gold_answers, samples)
    
    # 분석 결과 저장
    analysis_file = f"{analysis_dir}/incorrect_samples_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save_analysis_results(incorrect_data, analysis_file)
    
    # 오류 분포 시각화
    visualize_error_distribution(incorrect_data, analysis_dir, dataset_name)
    
    logging.info("분석 완료")

if __name__ == "__main__":
    main()