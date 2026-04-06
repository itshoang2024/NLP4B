# 1. Output schema chung
Nhánh agent nên trả candidate theo format ổn định:
```json
{
    "video_id": str,
    "frame_id": int,
    "branch": "agentic",
    "agent_score": float,
    "evidence": list[str],
    "trace": dict
}
```
Nhánh kia cũng nên có schema tương tự, chỉ đổi branch và manual_score.

# 2. Score direction

Phải thống nhất:
- score càng lớn càng tốt
- score đã normalize hay chưa
- final merger có dùng score thô hay calibrated score

# 3. Frame identity
Phải thống nhất cứng:
- video_id
- frame_id
để merge cuối không lỗi.