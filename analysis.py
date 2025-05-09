import json

# 경로 설정 (사용자 환경에 맞게 수정)
openai_triple_path = "outputs/musique/openie_results_ner_gpt-4o-mini.json"
llama_triple_path = "outputs/musique/openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json"

# 파일 로드
with open(openai_triple_path, 'r', encoding='utf-8') as f:
    openai_triple_json = json.load(f)
    
with open(llama_triple_path, 'r', encoding='utf-8') as f:
    llama_triple_json = json.load(f)

def compare_triples(openai_data, llama_data):
    """
    openai_data['docs']와 llama_data['docs']를 순회하며,
    각 문서별로 triple 비교 및 간단한 지표를 출력합니다.
    """
    
    openai_docs = openai_data.get("docs", [])
    llama_docs = llama_data.get("docs", [])
    
    # 두 모델의 문서 개수 중 최소값 기준으로만 비교
    num_docs = min(len(openai_docs), len(llama_docs))
    
    # 전체 통계를 위해 누적할 변수
    total_openai_triples = 0
    total_llama_triples = 0
    total_intersect_triples = 0
    
    for i in range(num_docs):
        openai_doc = openai_docs[i]
        llama_doc = llama_docs[i]
        
        # extracted_triples가 없으면 빈 리스트로 처리
        openai_triples_list = openai_doc.get("extracted_triples", [])
        llama_triples_list = llama_doc.get("extracted_triples", [])
        
        # 세트로 변환 (튜플화)하여 비교
        # 예: ["Lionel Messi", "enrolled in", "RFEF"]
        # 세 요소가 정확히 일치해야 동등하다고 봄
        openai_triples = set(tuple(t) for t in openai_triples_list)
        llama_triples  = set(tuple(t) for t in llama_triples_list)
        
        # 교집합, 차집합
        intersect = openai_triples.intersection(llama_triples)
        only_openai = openai_triples - llama_triples
        only_llama  = llama_triples - openai_triples
        
        # Precision, Recall, F1 계산(간단 버전)
        #  - Precision = (교집합 / Llama가 추출한 총 트리플 수)
        #  - Recall    = (교집합 / OpenAI가 추출한 총 트리플 수)
        #  - F1        = 2 * Precision * Recall / (Precision + Recall)
        
        openai_count = len(openai_triples)
        llama_count = len(llama_triples)
        intersect_count = len(intersect)
        
        precision = intersect_count / llama_count if llama_count else 0
        recall = intersect_count / openai_count if openai_count else 0
        f1 = 0
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        
        # 통계 누적
        total_openai_triples += openai_count
        total_llama_triples += llama_count
        total_intersect_triples += intersect_count
        
        print(f"[DOC {i}] Title: {openai_doc.get('title', 'N/A')} vs {llama_doc.get('title', 'N/A')}")
        print(f"  - OpenAI Triples: {openai_count}")
        print(f"  - Llama  Triples: {llama_count}")
        print(f"  - Intersection  : {intersect_count}")
        print(f"  - Only OpenAI   : {len(only_openai)}")
        print(f"  - Only Llama    : {len(only_llama)}")
        print(f"  - Precision     : {precision:.4f}")
        print(f"  - Recall        : {recall:.4f}")
        print(f"  - F1            : {f1:.4f}")
        print(f"  - Only OpenAI Triples (예시): {only_openai}")
        print(f"  - Only Llama  Triples (예시): {only_llama}")
        print("-" * 60)
    
    # 전체 문서 기준 통계
    print("\n=== 전체 문서 통계 ===")
    print(f"OpenAI가 추출한 총 트리플: {total_openai_triples}")
    print(f"Llama가 추출한 총 트리플: {total_llama_triples}")
    print(f"교집합(중복X) 트리플 총합: {total_intersect_triples}")
    
    # 간단한 micro-average 방식 계산(모든 문서의 트리플을 합쳐서 계산)
    micro_precision = total_intersect_triples / total_llama_triples if total_llama_triples else 0
    micro_recall = total_intersect_triples / total_openai_triples if total_openai_triples else 0
    micro_f1 = 0
    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall   : {micro_recall:.4f}")
    print(f"Micro F1       : {micro_f1:.4f}\n")

# 비교 실행
compare_triples(openai_triple_json, llama_triple_json)


'''
import json

openai_triple_path = "outputs/musique/openie_results_ner_gpt-4o-mini.json"
llama_triple_path = "outputs/musique/openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json"

# Open and load JSON files correctly
with open(openai_triple_path, 'r', encoding='utf-8') as file:
    openai_triple_json = json.load(file)

with open(llama_triple_path, 'r', encoding='utf-8') as file:
    llama_triple_json = json.load(file)

# llama_triple_json keys
# dict_keys(['docs', 'ents_by_doc', 'avg_ent_chars', 'avg_ent_words', 'num_tokens', 'approx_total_tokens'])

print("docs", llama_triple_json["docs"][0], end="\n\n")

print("docs keys", llama_triple_json["docs"][0].keys(), end="\n\n")
print("ents_by_doc", llama_triple_json["ents_by_doc"][0], end="\n\n")
print("avg_ent_chars", llama_triple_json["avg_ent_chars"], end="\n\n")
print("avg_ent_words", llama_triple_json["avg_ent_words"], end="\n\n")
print("num_tokens", llama_triple_json["num_tokens"], end="\n\n")
print("approx_total_tokens", llama_triple_json["approx_total_tokens"], end="\n\n")
'''