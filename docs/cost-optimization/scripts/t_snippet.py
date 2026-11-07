"""Chạy đúng đoạn 'Cách dùng' trong README với SDK thật và transport giả."""
import re, json
from decimal import Decimal
from pathlib import Path
import anthropic, httpx2
from budget import SpendLedger, BudgetExceeded, max_request_cost
from cost import request_cost

src = (Path(__file__).resolve().parent.parent / 'README.md').read_text()
snippet = next(b for b in re.findall(r"```python\n(.*?)```", src, re.S) if "ledger = SpendLedger" in b)

def run(status):
    def handler(req):
        if req.url.path.endswith("count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1000})
        if status == 200:
            return httpx2.Response(200, json={"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1000, "output_tokens": 10}})
        return httpx2.Response(status, json={"type": "error", "error": {"type": "x", "message": "m"}})
    client = anthropic.Anthropic(api_key="k", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    g = {"client": client, "model": "claude-sonnet-5-5", "system": "s", "messages": [{"role": "user", "content": "x"}],
         "max_tokens": 100, "SpendLedger": SpendLedger, "BudgetExceeded": BudgetExceeded, "max_request_cost": max_request_cost,
         "request_cost": request_cost, "Decimal": Decimal}
    try:
        exec(snippet, g)
    except anthropic.APIStatusError:
        pass
    return sum(g["ledger"].spent.values()), max_request_cost("claude-sonnet-5-5", 1000, 100)

spent, cap = run(200); assert spent == Decimal("0.0021"), spent          # (1000*2 + 10*10) / 1e6
spent, cap = run(400); assert spent == 0, spent
spent, cap = run(500); assert spent == cap, (spent, cap)
print("SNIPPET OK")
