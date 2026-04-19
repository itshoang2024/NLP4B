import os
import pandas as pd
import ast
import time
import numpy as np
import argparse
from dotenv import load_dotenv # Thêm thư viện này
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load các biến môi trường từ file .env
load_dotenv()


'''
python inference_baseline.py --csv_path "filtered_longvale_annotations.csv"  
                    --output_file ket_qua_search.json 
                    --id_start 0 
                    --id_end 3   
                    --k 5
'''

def check_overlap(gt_start, gt_end, ret_start, ret_end):
    """Kiểm tra xem timestamp của frame tìm được có nằm trong khoảng ground truth không"""
    return max(gt_start, ret_start) <= min(gt_end, ret_end)

def evaluate_longvale_pipeline(csv_path, qdrant_client, model, collection_name, k=5, id_start=0, id_end=None):
    df = pd.read_csv(csv_path)
    
    if id_end is None or id_end > len(df):
        id_end = len(df)
        
    df_subset = df.iloc[id_start:id_end]
    
    total_latency = 0
    hits_at_k = 0
    mrr_scores = []
    
    print(f"Bắt đầu chạy evaluation từ index {id_start} đến {id_end-1} ({len(df_subset)} queries)...\n")

    for idx, row in df_subset.iterrows():
        query_text = row['sentences']
        gt_video_id = row['video_id']
        
        try:
            gt_start, gt_end = ast.literal_eval(row['time_stamp'])
        except:
            print(f"Lỗi parse timestamp ở index {idx}, bỏ qua.")
            continue

        # ==========================================
        # INFERENCE & ĐO LATENCY THẬT
        # ==========================================
        start_time = time.perf_counter()
        
        query_vector = model.encode(query_text).tolist()
        
        try:
            # Ưu tiên dùng hàm API mới nhất của Qdrant
            if hasattr(qdrant_client, 'query_points'):
                response = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using="keyframe-dense",
                    limit=k,
                    with_payload=True 
                )
                search_results = response.points
            else:
                # Fallback về hàm search cho bản cũ
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=("keyframe-dense", query_vector),
                    limit=k,
                    with_payload=True 
                )
        except Exception as e:
            print(f"Lỗi truy vấn Qdrant tại index {idx}: {e}")
            continue
            
        latency = time.perf_counter() - start_time
        total_latency += latency

        # ==========================================
        # TRÍCH XUẤT FRAME_ID TỪ KẾT QUẢ THẬT
        # ==========================================
        top_k_frame_ids = []
        is_hit = 0
        best_rank = float('inf')
        
        for rank_idx, res in enumerate(search_results):
            payload = res.payload
            
            ret_video_id = payload.get("video_id")
            ret_start = payload.get("timestamp_start")
            ret_end = payload.get("timestamp_end")
            frame_idx = payload.get("frame_idx")
            
            frame_id = f"{ret_video_id}_{frame_idx}"
            top_k_frame_ids.append(frame_id)

            if ret_video_id == gt_video_id and check_overlap(gt_start, gt_end, ret_start, ret_end):
                is_hit = 1
                best_rank = min(best_rank, rank_idx + 1)

        hits_at_k += is_hit
        mrr_scores.append(1.0 / best_rank if is_hit else 0.0)
        
        print(f"[{idx}] Query: {query_text[:40]}...")
        print(f"    Top-{k} Frames: {top_k_frame_ids} | Latency: {latency:.4f}s")
        print("-" * 50)

    # ==========================================
    # TỔNG KẾT METRIC
    # ==========================================
    if len(df_subset) > 0:
        final_avg_latency = total_latency / len(df_subset)
        final_hr_k = hits_at_k / len(df_subset)
        final_mrr = np.mean(mrr_scores)

        print("\n" + "="*45)
        print(f"📊 KẾT QUẢ ĐÁNH GIÁ (INDEX {id_start} -> {id_end-1})")
        print("="*45)
        print(f"Tổng số queries: {len(df_subset)}")
        print(f"Avg Latency    : {final_avg_latency:.4f} giây/query")
        print(f"HitRatio@{k}    : {final_hr_k:.4f}")
        print(f"MRR            : {final_mrr:.4f}")
        
        return final_hr_k, final_mrr
    else:
        print("Không có dữ liệu nào được chạy.")
        return 0, 0

# ==========================================
# CẤU HÌNH ARGPARSE 
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script cho Qdrant thật với URL & API Key từ .env")
    
    parser.add_argument("--csv_path", type=str, required=True, help="Đường dẫn đến file dataset")
    parser.add_argument("--k", type=int, default=5, help="Số lượng Top-K kết quả cần lấy (mặc định: 5)")
    parser.add_argument("--id_start", type=int, default=0, help="Index bắt đầu chạy (mặc định: 0)")
    parser.add_argument("--id_end", type=int, default=None, help="Index kết thúc")
    
    # Bỏ required=True, thay default bằng biến môi trường lấy từ .env
    parser.add_argument("--qdrant_url", type=str, default=os.getenv("QDRANT_URL"), help="URL của Qdrant server")
    parser.add_argument("--qdrant_api_key", type=str, default=os.getenv("QDRANT_API_KEY"), help="API Key của Qdrant")
    parser.add_argument("--collection_name", type=str, default=os.getenv("COLLECTION_NAME", "keyframes_v1"), help="Tên collection trên Qdrant")

    args = parser.parse_args()

    # Kiểm tra xem đã có URL và API Key chưa (từ file .env hoặc command line)
    if not args.qdrant_url or not args.qdrant_api_key:
        print("LỖI: Chưa cấu hình QDRANT_URL hoặc QDRANT_API_KEY trong file .env hoặc tham số truyền vào.")
        exit(1)

    try:
        print(f"Đang kết nối tới Qdrant tại {args.qdrant_url}...")
        qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
        
        print("Đang load model SigLIP...")
        siglip_model = SentenceTransformer('google/siglip-so400m-patch14-384')
        print("Khởi tạo hoàn tất!\n")
    except Exception as e:
        print(f"Lỗi khởi tạo kết nối hoặc model: {e}")
        exit(1)

    evaluate_longvale_pipeline(
        csv_path=args.csv_path,
        qdrant_client=qdrant,
        model=siglip_model,
        collection_name=args.collection_name,
        k=args.k,
        id_start=args.id_start,
        id_end=args.id_end
    )