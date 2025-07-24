# ------------------------------------------
MODEL_PRICE = {
    'gpt-4o-mini': {"input_token": 0.15, "output_token": 0.60},
    'claude-3-haiku': {"input_token": 0.25, "output_token": 1.25},
    'gemini-2.0-flash': {"input_token": 0.1, "output_token": 0.4},
}

""" 전처리 시(필터링, 클리닝), 모델별 입력 토큰 사용량"""
avg_input_token_F = {
    "gpt-4o-mini": 558,
    "gemini-2.0-flash": 513 
}

avg_output_token_F = {
    "gpt-4o-mini": 1,
    "gemini-2.0-flash": 1,
}

avg_input_token_C = {
    "gpt-4o-mini": 695,
    "gemini-2.0-flash": 644,
}

avg_output_token_C = {
    "gpt-4o-mini": 218,
    "gemini-2.0-flash": 577,
}

""" 답변 생성 시, 모델별 입력 토큰 사용량
- gpt-4o-mini: 약 300 토큰
- claude-3-haiku: 약 500 토큰
- gemini-2.0-flash: 약 280 토큰
"""
avg_input_token_A = {
    "0": {"gpt-4o-mini": 265, "claude-3-haiku": 417, "gemini-2.0-flash": 231},
    "1": {"gpt-4o-mini": 567, "claude-3-haiku": 916, "gemini-2.0-flash": 510},
    "3": {"gpt-4o-mini": 1181, "claude-3-haiku": 1933, "gemini-2.0-flash": 1074},
    "6": {"gpt-4o-mini": 2106, "claude-3-haiku": 3464, "gemini-2.0-flash": 1925},
}
""" 답변 생성 시, 모델별 출력 토큰 사용량"""
avg_output_token_A = {
    "gpt-4o-mini": 127,
    "claude-3-haiku": 238,
    "gemini-2.0-flash": 122,
}

# ------------------------------------------
PREPROCESS_MODEL = "gemini-2.0-flash"
MILLION_TOKEN = 1000000

FILTERING_SIZE = 61825
CLEANING_SIZE = 53425
TEST_SIZE = 10000
# ------------------------------------------
print(f"전처리 모델: {PREPROCESS_MODEL}")
# ------------------------------------------
filtering_cost = MODEL_PRICE[PREPROCESS_MODEL]["input_token"] * avg_input_token_F[PREPROCESS_MODEL]+ MODEL_PRICE[PREPROCESS_MODEL]["output_token"] * avg_output_token_F[PREPROCESS_MODEL]
filtering_cost = (filtering_cost * FILTERING_SIZE) / MILLION_TOKEN
print(f"필터링 비용: ${filtering_cost:.2f}")
# ------------------------------------------
cleaning_cost = MODEL_PRICE[PREPROCESS_MODEL]["input_token"] * avg_input_token_C[PREPROCESS_MODEL] + MODEL_PRICE[PREPROCESS_MODEL]["output_token"] * avg_output_token_C[PREPROCESS_MODEL]
cleaning_cost = (cleaning_cost * CLEANING_SIZE) / MILLION_TOKEN
print(f"클리닝 비용: ${cleaning_cost:.2f}")
print("---"*10)
# ------------------------------------------
print("답변 생성 비용 (입력 형식 고려)")
for model_name, price in MODEL_PRICE.items():
    for shot, input_token_dict in avg_input_token_A.items():
        answering_cost = price["input_token"] * input_token_dict[model_name] + price["output_token"] * avg_output_token_A[model_name]
        answering_cost = (answering_cost * TEST_SIZE) / MILLION_TOKEN * 2  # 입력 형식 고려
        print(f"{model_name}-{shot}: ${answering_cost:.2f}")
    print("---"*10)
