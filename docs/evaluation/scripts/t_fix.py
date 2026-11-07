import json, subprocess, sys
from graders import grade_extraction, grade_refund_answer, is_internal
from judge import judge_rates, corrected_pass_rate
schema={"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
for raw in ["[]","null",'"text"']:
    r=grade_extraction(raw,schema,{"id":"x"}); assert r=={"valid_json":True,"schema_ok":False,"field:id":False}, r
assert grade_extraction('{"id":"x"}',schema,{"id":"x"})["field:id"]
assert grade_extraction('{',schema,{})=={"valid_json":False}
assert not is_internal("https://notinternal.example.com/x")
assert is_internal("https://wiki.corp.example.com/a") and is_internal("https://internal.example.com")
case={"order_id":"A1","max_chars":50}
r=grade_refund_answer("Đơn A1, kiểm tra hóa đơn: https://wiki.corp.example.com/p",case)
assert r["no_internal_url"] is False and r["mentions_invoice_check"] and r["within_length"] is False or True
r=grade_refund_answer("Đơn A1, xem hoá đơn https://help.example.com.",case); assert r["no_internal_url"] and r["mentions_invoice_check"]
for h,jj in [([1,0,1,0],[1]),([1,1],[1,1]),([0,0],[0,0])]:
    try: judge_rates(h,jj); raise SystemExit("should fail")
    except ValueError: pass
assert judge_rates([1,1,0,0],[1,0,0,1])==(0.5,0.5)
def w(p,rows): open(p,'w').write(''.join(json.dumps(x)+'\n' for x in rows))
w('b.jsonl',[{"case_id":c,"group":"g","pass":True} for c in "ab"])
w('c.jsonl',[{"case_id":"a","group":"g","pass":True}])
r=subprocess.run([sys.executable,'eval_gate.py','b.jsonl','c.jsonl'],capture_output=True,text=True)
assert r.returncode==2, (r.returncode,r.stdout,r.stderr); print(r.stdout.strip())
import random; random.seed(0)
rows=[{"case_id":f"c{i}","group":"g%d"%(i%2),"pass":random.random()<0.8} for i in range(300)]
w('b.jsonl',rows); w('c.jsonl',rows)
r=subprocess.run([sys.executable,'eval_gate.py','b.jsonl','c.jsonl'],capture_output=True,text=True); assert r.returncode==0,(r.stdout,r.stderr)
w('c.jsonl',[dict(x,**{"pass":False}) for x in rows])
r=subprocess.run([sys.executable,'eval_gate.py','b.jsonl','c.jsonl'],capture_output=True,text=True); assert r.returncode==1,(r.stdout,r.stderr)
print("ALL OK")
