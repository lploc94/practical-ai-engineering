import asyncio, json, time, httpx, remote_server as rs
TOKENS = {"good": {"client_id":"c1","scope":"orders:read","exp":int(time.time())+3600,"aud":rs.RESOURCE,"sub":"u42"},
          "otheraud": {"client_id":"c1","scope":"orders:read","exp":int(time.time())+3600,"aud":"https://other.example/mcp","sub":"u42"},
          "noscope": {"client_id":"c1","scope":"profile","exp":int(time.time())+3600,"aud":rs.RESOURCE,"sub":"u42"}}
rs.decode_and_verify_jwt = TOKENS.get
rs.orders_of = lambda u: [f"{u}-A1001"]
M = {"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}
async def main():
    async with rs.app.router.lifespan_context(rs.app):
        t = httpx.ASGITransport(app=rs.app)
        async with httpx.AsyncClient(transport=t, base_url="https://mcp.shop.example") as c:
            def h(tok=None, **extra):
                d={"Content-Type":"application/json","Accept":"application/json, text/event-stream","MCP-Protocol-Version":"2026-07-28","Mcp-Method":"tools/call","Mcp-Name":"my_orders",**extra}
                if tok: d["Authorization"]=f"Bearer {tok}"
                return d
            body={"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"my_orders","arguments":{},"_meta":M}}
            for tok in (None,"bad","otheraud","noscope","good"):
                r = await c.post("/mcp", headers=h(tok), json=body)
                print(tok, r.status_code, r.headers.get("www-authenticate"), r.text[:200] if r.status_code==200 else "")
            r = await c.post("/mcp", headers=h("good", Origin="https://evil.example"), json=body); print("evil origin", r.status_code)
            r = await c.get("/.well-known/oauth-protected-resource/mcp"); print("PRM", r.status_code, r.text)
asyncio.run(main())
