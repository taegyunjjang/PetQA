from time import sleep
import json
import argparse
from trafilatura import fetch_url, extract
from tqdm import tqdm
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    save_json, load_json, setup_logging, load_environment
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()


MAX_RETRIES = 3


def url2text(url):
    downloaded = None
    num_fails = 0
    while not downloaded and num_fails < MAX_RETRIES:
        try:
            downloaded = fetch_url(url)
        except Exception as e:
            num_fails += 1
            sleep(3)
    
    result = extract(downloaded, output_format="json", include_comments=False)
    result = json.loads(result)
    if result is not None:
        return result["text"]
    else:
        print(f"Failed to extract text from {url}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    env = load_environment()
    logger = setup_logging()

    input_path = env["serper_results_path"]
    output_path = env["crawled_results_path"]
    data = load_json(input_path)

    query_order = []
    results_by_query = {}
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_info = {}
        for item in tqdm(data, desc="Processing data"):
            query = item["searchParameters"]["q"]
            
            if query not in results_by_query:
                query_order.append(query)
                results_by_query[query] = []
                
            for org in item["organic"]:
                link = org["link"]
                position = org["position"]
                future = executor.submit(url2text, link)
                future_to_info[future] = {
                    "query": query,
                    "link": link,
                    "position": position
                }
    
        for future in tqdm(as_completed(future_to_info), 
                        total=len(future_to_info), desc="Processing results"):
            info = future_to_info[future]
            text_result = future.result()
            results_by_query[info["query"]].append({
                "link": info["link"],
                "content": text_result,
                "position": info["position"]
            })
            
    final_results = []
    for query in query_order:
        urls_data = results_by_query[query]
        sorted_urls = sorted(urls_data, key=lambda x: x["position"])
        final_results.append({
            "query": query,
            "urls": sorted_urls
        })
    
    save_json(final_results, output_path)
    logger.info(f"Saved {len(final_results)} queries to {output_path}")
            
            
if __name__ == "__main__":
    main()