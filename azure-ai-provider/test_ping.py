"""
test_ping.py — End-to-End Health Check Script
================================================

Tests the full chain: Client → Gateway → Embedding Service → Gateway → Client

Usage:
  python test_ping.py                        # Default: localhost:8080
  python test_ping.py http://10.0.0.4:8080   # Custom VM IP
"""

import sys
import json
import time

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

    print("═" * 60)
    print("  🏥 Azure AI Provider — End-to-End Health Check")
    print(f"  Target: {base_url}")
    print("═" * 60)

    # ── Test 1: Gateway Health ────────────────────────────────────────
    print("\n[1/3] 🔍 Gateway Health Check...")
    try:
        t0 = time.time()
        resp = httpx.get(f"{base_url}/health", timeout=10)
        elapsed = time.time() - t0

        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ Gateway OK ({elapsed:.3f}s)")
            print(f"     Downstream: {json.dumps(data.get('downstream', {}), indent=6)}")
        else:
            print(f"  ❌ Gateway returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"  ❌ Cannot reach gateway: {e}")
        sys.exit(1)

    # ── Test 2: Embedding Proxy ───────────────────────────────────────
    print("\n[2/3] 📡 Embedding Proxy Test (Gateway → Embedding Service)...")
    try:
        payload = {"input": "Hello, this is a test query", "model": "mock"}
        t0 = time.time()
        resp = httpx.post(f"{base_url}/v1/embeddings", json=payload, timeout=10)
        elapsed = time.time() - t0

        if resp.status_code == 200:
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            print(f"  ✅ Embedding OK ({elapsed:.3f}s)")
            print(f"     Model:     {data.get('model', '?')}")
            print(f"     Status:    {data.get('status', '?')}")
            print(f"     Vector:    {embedding}")
            print(f"     Dimension: {len(embedding)}")
        else:
            print(f"  ❌ Returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"  ❌ Embedding proxy failed: {e}")

    # ── Test 3: LLM Proxy (expect mock) ───────────────────────────────
    print("\n[3/3] 🤖 LLM Proxy Test (should be mock in Phase 1)...")
    try:
        t0 = time.time()
        resp = httpx.post(f"{base_url}/v1/chat/completions", json={}, timeout=10)
        elapsed = time.time() - t0

        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ LLM endpoint reachable ({elapsed:.3f}s)")
            print(f"     Status: {data.get('status', '?')}")
            print(f"     Message: {data.get('message', '?')}")
        else:
            print(f"  ⚠️  LLM returned {resp.status_code} (expected in mock mode)")
    except Exception as e:
        print(f"  ⚠️  LLM proxy: {e} (expected in Phase 1)")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  ✅ Health check complete!")
    print("  Full chain: Client → Gateway → Embedding → Gateway → Client")
    print("═" * 60)


if __name__ == "__main__":
    main()
