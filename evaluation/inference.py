import pandas as pd
import ast
import time
import numpy as np
import argparse
import json
import requests

'''
python inference.py --csv_path "filtered_longvale_annotations.csv"  
                    --endpoint search/heuristic # search | search/heuristic | search/agentic
                    --output_file ket_qua_search.json 
                    --id_start 0 
                    --id_end 3   
                    --k 5
'''


def check_overlap(gt_start, gt_end, ret_start, ret_end):
    if ret_start is None or ret_end is None:
        return False
    return max(gt_start, ret_start) <= min(gt_end, ret_end)

def run_api_inference(csv_path, base_url, endpoint, k=5, id_start=0, id_end=None, output_file="api_metrics.json"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    if id_end is None or id_end > len(df):
        id_end = len(df)
    df_subset = df.iloc[id_start:id_end]
    
    hits_at_k = 0
    mrr_scores = []
    detailed_results = []
    
    api_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    print(f"🚀 Evaluation: {api_url} | K={k}")

    for idx, row in df_subset.iterrows():
        query_text = row['sentences']
        gt_vid = row['video_id']
        try:
            gt_ts = ast.literal_eval(row['time_stamp']) # [start, end]
        except:
            continue

        start_time = time.perf_counter()
        try:
            # Gửi đúng payload raw_query mà backend yêu cầu
            response = requests.post(api_url, json={"raw_query": query_text, "top_k": k})
            response.raise_for_status()
            api_data = response.json()
        except Exception as e:
            print(f"[{idx}] ❌ API Error: {e}")
            continue
            
        latency = time.perf_counter() - start_time

        # --- XỬ LÝ KẾT QUẢ ---
        # Kiểm tra kỹ cấu trúc trả về của Backend (results hoặc data)
        search_results = api_data if isinstance(api_data, list) else (api_data.get("results") or api_data.get("data") or [])
        
        top_k_info = []
        is_hit = 0
        best_rank = float('inf')
        
        print(f"\n[Index {idx}] Query: {query_text[:60]}...")
        print(f"  📍 GT: {gt_vid} | Timestamp: {gt_ts}")

        if not search_results:
            print("  ⚠️ WARNING: API trả về mảng rỗng [] - Kiểm tra lại database/collection!")

        for rank_idx, res in enumerate(search_results[:k]):
            ret_vid = res.get("video_id")
            ret_start = res.get("timestamp_start") or res.get("start")
            ret_end = res.get("timestamp_end") or res.get("end")
            f_idx = res.get("frame_idx") or 0
            
            match = (ret_vid == gt_vid and check_overlap(gt_ts[0], gt_ts[1], ret_start, ret_end))
            
            if match:
                is_hit = 1
                best_rank = min(best_rank, rank_idx + 1)
            
            res_detail = {
                "rank": rank_idx + 1,
                "video_id": ret_vid,
                "timestamp": [ret_start, ret_end],
                "match": match
            }
            top_k_info.append(res_detail)
            
            # In ra màn hình để ông soi độ lệch
            status_icon = "✅" if match else "❌"
            print(f"  {status_icon} Rank {rank_idx+1}: {ret_vid} | TS: [{ret_start}, {ret_end}]")

        # Update metrics
        hits_at_k += is_hit
        mrr_scores.append(1.0 / best_rank if is_hit else 0.0)
        
        detailed_results.append({
            "index": idx,
            "ground_truth": {"video_id": gt_vid, "timestamp": gt_ts},
            "retrieved": top_k_info,
            "latency": round(latency, 4)
        })

    # Xuất file tổng kết
    final_report = {
        "metrics": {"HitRatio@5": np.mean([r > 0 for r in mrr_scores]), "MRR": np.mean(mrr_scores)},
        "details": detailed_results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="https://nlp4b.vercel.app")
    parser.add_argument("--endpoint", type=str, required=True, help="Điền endpoint (vd: search, heuristic, agentic)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--id_start", type=int, default=0)
    parser.add_argument("--id_end", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="api_metrics.json")
    
    args = parser.parse_args()

    run_api_inference(
        csv_path=args.csv_path,
        base_url=args.base_url,
        endpoint=args.endpoint,
        k=args.k,
        id_start=args.id_start,
        id_end=args.id_end,
        output_file=args.output_file
    )