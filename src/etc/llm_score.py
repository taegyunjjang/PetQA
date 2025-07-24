import json
from collections import defaultdict


# 지표 목록
metrics = ["factuality", "completeness", "coherence"]

# 첫 번째 파일: 평균 점수 계산
def compute_average_scores(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    score_sums = defaultdict(int)
    count = defaultdict(int)

    for entry in data:
        for metric in metrics:
            if metric in entry and "score" in entry[metric]:
                score_sums[metric] += entry[metric]["score"]
                count[metric] += 1

    average_scores = {metric: round(score_sums[metric] / count[metric], 3) if count[metric] > 0 else None
                      for metric in metrics}
    return average_scores


model_name = ["gpt-4o-mini", "gpt-4.1-nano", "claude-3-haiku", "gemini-2.0-flash"]
shot = [0, 1, 3, 6]
suffix = "_raw"
for m in model_name:
    for s in shot:
        score_file_path = f"./data/llm_eval/score_results_{m}_{s}{suffix}.json"
        pairwise_file_path = f"./data/llm_eval/pairwise_results_{m}_{s}{suffix}.json"

        print(f"model: {m}")
        print(f"shot: {s}")
        print(compute_average_scores(score_file_path))
        print('-'*50)
        
    print()
    print('-'*50)