import anthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import Client

ALLOWED = {"get_order"}               # tool ghi như refund_order không đưa thẳng vào runner


async def ask(server, question: str) -> str:
    client = anthropic.AsyncAnthropic()
    # server: một MCPServer (chạy trong process), URL "https://.../mcp", hoặc StdioServerParameters
    async with Client(server) as mcp_client:
        listed = await mcp_client.list_tools()
        tools = [async_mcp_tool(t, mcp_client) for t in listed.tools if t.name in ALLOWED]
        runner = client.beta.messages.tool_runner(
            model="claude-sonnet-5-5", max_tokens=4096, tools=tools,
            messages=[{"role": "user", "content": question}])
        final = await runner.until_done()
        return "".join(b.text for b in final.content if b.type == "text")
