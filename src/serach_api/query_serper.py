import requests
from tqdm import tqdm
import sys
import os
import time
import random
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    save_json, load_json, setup_logging, load_environment
)
from dotenv import load_dotenv
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

_SERPER_URL = "https://google.serper.dev/search"
_EXCLUDE_URL = "kin.naver.com/"
MAX_RETRIES = 20

class SerperAPI:
    def __init__(
        self,
        serper_api_key=SERPER_API_KEY,
        gl="kr",  # country
        hl="ko",  # language
        num=3,    # number of results
        search_type="search",
    ):
        self.serper_api_key = serper_api_key
        self.gl = gl
        self.hl = hl
        self.num = num
        self.search_type = search_type


    def run(self, query):
        """Run query through GoogleSearch and parse results."""
        assert self.serper_api_key, 'Missing serper_api_key'
        results = self._google_serper_api_results(
            query,
            gl = self.gl,
            hl = self.hl,
            num = self.num,
            search_type = self.search_type,
        )
        return results
    
    
    def _google_serper_api_results(
        self,
        query,
        **kwargs,
    ):
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        
        full_query = f"{query} -site:{_EXCLUDE_URL}"
        payload = {
            "q": full_query,
            **kwargs,
        }
        response, num_fails, sleep_time = None, 0, 0
        
        while not response and num_fails < MAX_RETRIES:
            try:
                response = requests.request("POST", _SERPER_URL, headers=headers, json=payload)
            except AssertionError as e:
                raise e
            except Exception:
                response = None
                num_fails += 1
                sleep_time = random.uniform(1, 3) * (2 ** num_fails)
                sleep_time = min(sleep_time, 600)
                time.sleep(sleep_time)
        
        if not response:
            raise ValueError(f'Failed to get results from Google Serper API')
        
        response.raise_for_status()
        search_results = response.json()
        return search_results
    
    
def main():
    env = load_environment()
    logger = setup_logging()
    test_data = load_json(env["data_files"]["test"])
    serper_searcher = SerperAPI()
    
    output_path = env["serper_results_path"]
    results = []
    batch_size = 100
    current_batch = []
    for i, item in enumerate(tqdm(test_data, total=len(test_data), desc="Serper API Query")):
        query = item["question"]
        result = serper_searcher.run(query)
        current_batch.append(result)
        
        if (i + 1) % batch_size == 0 or (i + 1) == len(test_data):
            results.extend(current_batch)
            save_json(results, output_path)
            logger.info(f"Saved batch {i//batch_size+1} ({i+1}/{len(test_data)})")
            current_batch = []
    
    logger.info(f"Serper results saved: {output_path}")

if __name__ == "__main__":
    main()