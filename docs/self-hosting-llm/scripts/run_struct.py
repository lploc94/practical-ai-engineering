"""Cùng model, cùng prompt: có và không có response_format json_schema. Đếm câu trả lời
không parse được JSON, không đúng schema, và exact match."""
import asyncio
import json
import sys

import httpx2

from run_quant import URL, start

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["intent", "order_id"],
          "properties": {"intent": {"enum": ["track", "cancel", "return", "warranty", "payment", "other"]},
                         "order_id": {"anyOf": [{"type": "string", "pattern": "^DH-[0-9]{6}$"}, {"type": "null"}]}}}


def valid(obj):
    return (isinstance(obj, dict) and set(obj) == {"intent", "order_id"}
            and obj["intent"] in SCHEMA["properties"]["intent"]["enum"]
            and (obj["order_id"] is None or (isinstance(obj["order_id"], str) and obj["order_id"].startswith("DH-"))))


async def score(rows, fmt):
    sem = asyncio.Semaphore(8)
    out = {"bad_json": 0, "bad_schema": 0, "exact": 0}
    async with httpx2.AsyncClient(timeout=120) as c:
        async def one(r):
            async with sem:
                body = {"model": "m", "messages": r["messages"][:-1], "max_tokens": 64, "temperature": 0}
                if fmt:
                    body["response_format"] = {"type": "json_schema", "json_schema": {"name": "intent", "schema": SCHEMA}}
                text = (await c.post(URL, json=body)).json()["choices"][0]["message"]["content"]
            try:
                got = json.loads(text)
            except json.JSONDecodeError:
                out["bad_json"] += 1
                return
            if not valid(got):
                out["bad_schema"] += 1
            elif got == json.loads(r["messages"][-1]["content"]):
                out["exact"] += 1
        await asyncio.gather(*(one(r) for r in rows))
    return out


if __name__ == "__main__":
    rows, seen = [], set()
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        if r["messages"][1]["content"] not in seen:
            seen.add(r["messages"][1]["content"])
            rows.append(r)
    for m in sys.argv[2:]:
        p = start(m, "-c", "16384", "-np", "8")
        try:
            for fmt in (False, True):
                print(m.split("/")[-1], "json_schema" if fmt else "free", len(rows), asyncio.run(score(rows, fmt)), flush=True)
        finally:
            p.terminate()
            p.wait()
