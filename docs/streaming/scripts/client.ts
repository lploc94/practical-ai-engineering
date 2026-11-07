import { createParser, type EventSourceMessage } from "eventsource-parser";

export async function streamChat(body: unknown, signal: AbortSignal,
                                 onEvent: (e: EventSourceMessage) => void) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,                                    // nút "Dừng" gọi controller.abort()
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  let finished = false;
  let failure = null as { code: string; retryable: boolean } | null; // `as`: TS không theo dõi gán trong callback
  const parser = createParser({
    onEvent(e) {
      if (e.event === "done") finished = true;               // event kết thúc do backend định nghĩa (§3.1)
      else if (e.event === "error") failure = JSON.parse(e.data); // lỗi đã được backend phân loại
      onEvent(e);
    },
  });
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.feed(value);
  }
  parser.reset({ consume: true });                            // xử lý phần còn lại trong buffer
  if (failure) throw new StreamError(failure.code, failure.retryable);
  if (!finished) throw new StreamError("stream_truncated", true); // đóng kết nối mà không có done hay error
}

export class StreamError extends Error {
  constructor(public code: string, public retryable: boolean) { super(code); }
}
