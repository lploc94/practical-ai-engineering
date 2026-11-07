import ast
import asyncio
import re

import anthropic
import httpx2

from policy_check import make_policy_check
from stream_guard import guarded_text
from t_stream_guard import events
from t_policy_check import classifier_client, label, msg

src = open(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'README.md'), encoding="utf-8").read()
snippet = re.search(r"Cách dùng với policy check ở §2.5:\n\n```python\n(.*?)```", src, re.S).group(1)
code = compile(snippet, "usage", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def main_client(body):
    def handler(request):
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    return anthropic.AsyncAnthropic(api_key="k", max_retries=0,
                                    http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


async def run(body):
    sent = []
    async def send_to_client(t): sent.append(("text", t))
    async def send_to_client_replace(t): sent.append(("replace", t))
    async def retract_and_explain(r): sent.append(("retract", r))
    ns = dict(make_policy_check=make_policy_check, guarded_text=guarded_text, SAFE_REPLY="SAFE",
              classifier_client=classifier_client(lambda t: msg(label("refund_promise" if "hoàn tiền" in t else "ok"))),
              client=main_client(body), messages=[{"role": "user", "content": "hi"}],
              send_to_client=send_to_client, send_to_client_replace=send_to_client_replace,
              retract_and_explain=retract_and_explain)
    await eval(code, ns)
    return sent


async def main():
    s1 = "Đơn của anh đang được giao tới nơi rồi ạ, bên vận chuyển báo dự kiến tới vào thứ Sáu tuần này. "
    s2 = "Shop cam kết hoàn tiền 100% nếu trễ. "
    sent = await run(events([s1, s2]))
    assert sent == [("text", s1), ("replace", "SAFE")], sent
    sent = await run(events([s1, "dở"], stop="refusal",
                            details={"type": "refusal", "category": "cyber", "explanation": "x"}))
    assert sent == [("text", s1), ("retract", "cyber")], sent
    print("USAGE OK")


asyncio.run(main())
