import os
import json
import logging
import yaml


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"

def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    return logger

def save_json(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving {file_path}: {e}")

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def load_results(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"{start_idx}개까지 처리됨. 이어서 시작")
    else:
        results = []
        start_idx = 0
        print("새로 시작")
    return results, start_idx

def load_environment():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PROMPT_DIR = os.path.join(PROJECT_ROOT, 'prompts')
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    UTILS_DIR = os.path.join(SRC_DIR, 'utils')

    SYSTEM_FILTERING_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_filtering.txt')
    SYSTEM_CLEANING_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_cleaning.txt')
    USER_PREPROCESSING_PROMPT_PATH = os.path.join(PROMPT_DIR, 'user_preprocessing.txt')

    SYSTEM_ATOMIC_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_atomic.txt')
    USER_ATOMIC_PROMPT_PATH = os.path.join(PROMPT_DIR, 'user_atomic.txt')

    SYSTEM_ZEROSHOT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_zeroshot.txt')
    SYSTEM_FEWSHOT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_fewshot.txt')
    USER_PROCESSED_INPUT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'user_processed_input.txt')
    USER_RAW_INPUT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'user_raw_input.txt')

    SYSTEM_LLM_SCORE_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_llm_score.txt')
    USER_LLM_SCORE_PROMPT_PATH = os.path.join(PROMPT_DIR, 'user_llm_score.txt')

    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    RAW_DATA_PATH = os.path.join(INTERIM_DATA_DIR, 'unique_data.json')
    FILTERING_ALL_RESULTS_PATH = os.path.join(INTERIM_DATA_DIR, 'filtering_all_results.json')
    IRRELEVANT_DATA_PATH = os.path.join(INTERIM_DATA_DIR, 'irrelevant_data.json')
    FILTERED_DATA_PATH = os.path.join(INTERIM_DATA_DIR, 'filtered_data.json')
    CLEANED_DATA_PATH = os.path.join(INTERIM_DATA_DIR, 'cleaned_data.json')
    TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.json')
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    ATOMIC_FACTS_DIR = os.path.join(RESULTS_DIR, 'atomic_facts')
    GENERATED_ANSWERS_DIR = os.path.join(RESULTS_DIR, 'generated_answers')
    
    GOLD_FACTS_PATH = os.path.join(ATOMIC_FACTS_DIR, 'gold_facts.json')
    FEWSHOT_EXAMPLES_PATH = os.path.join(PROCESSED_DATA_DIR, 'fewshot_examples.json')
    
    DATA_FILES = {
        "train": os.path.join(PROCESSED_DATA_DIR, 'train.json'),
        "validation": os.path.join(PROCESSED_DATA_DIR, 'validation.json')
    }
    
    CONFIG_DIR = os.path.join(UTILS_DIR, 'config')
    CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.yaml')
    
    return {
        "system_filtering_prompt_path": SYSTEM_FILTERING_PROMPT_PATH,
        "system_cleaning_prompt_path": SYSTEM_CLEANING_PROMPT_PATH,
        "user_preprocessing_prompt_path": USER_PREPROCESSING_PROMPT_PATH,
        "system_atomic_prompt_path": SYSTEM_ATOMIC_PROMPT_PATH,
        "user_atomic_prompt_path": USER_ATOMIC_PROMPT_PATH,
        "system_zeroshot_prompt_path": SYSTEM_ZEROSHOT_PROMPT_PATH,
        "system_fewshot_prompt_path": SYSTEM_FEWSHOT_PROMPT_PATH,
        "user_processed_input_prompt_path": USER_PROCESSED_INPUT_PROMPT_PATH,
        "user_raw_input_prompt_path": USER_RAW_INPUT_PROMPT_PATH,
        "system_llm_score_prompt_path": SYSTEM_LLM_SCORE_PROMPT_PATH,
        "user_llm_score_prompt_path": USER_LLM_SCORE_PROMPT_PATH,
        "raw_data_path": RAW_DATA_PATH,
        "test_data_path": TEST_DATA_PATH,
        "filtering_all_results_path": FILTERING_ALL_RESULTS_PATH,
        "irrelevant_data_path": IRRELEVANT_DATA_PATH,
        "filtered_data_path": FILTERED_DATA_PATH,
        "cleaned_data_path": CLEANED_DATA_PATH,
        "checkpoint_dir": CHECKPOINT_DIR,
        "best_model_dir": BEST_MODEL_DIR,
        "atomic_facts_dir": ATOMIC_FACTS_DIR,
        "gold_facts_path": GOLD_FACTS_PATH,
        "generated_answers_dir": GENERATED_ANSWERS_DIR,
        "fewshot_examples_path": FEWSHOT_EXAMPLES_PATH,
        "data_files": DATA_FILES,
        "config_path": CONFIG_PATH,
    }



""" model name to endpoint mapping """
MODEL_MAPPING = {
    # closed LLM
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    
    # open LLM
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    
    # judge LLM
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "exaone-3.5-32b": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
}