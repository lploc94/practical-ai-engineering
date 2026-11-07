from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token


class SafeMarkdown:
    """Render markdown do model sinh ra thành HTML mà không để trình duyệt tự gửi request
    ra ngoài. Không render HTML thô; ảnh chỉ từ host trong `image_hosts`; link chỉ https tới
    host trong `link_hosts`, link khác thành chữ thường kèm host để người dùng tự quyết.
    Lọc trên token sau khi parse, nên bắt được cả link kiểu reference ([x][id]) và autolink."""

    def __init__(self, *, image_hosts: set[str], link_hosts: set[str]):
        self.image_hosts, self.link_hosts = image_hosts, link_hosts
        self.md = MarkdownIt("commonmark", {"html": False})

    @staticmethod
    def _host(url: str, allowed: set[str]) -> tuple[bool, str]:
        try:
            parts = urlsplit(url)
            host = (parts.hostname or "").lower()
        except ValueError:
            return False, "url lỗi"
        return parts.scheme == "https" and host in allowed, host or parts.scheme or "url lạ"

    def _filter(self, children: list[Token]) -> list[Token]:
        out, dropped = [], []                     # dropped: host của các link đang mở bị bỏ
        for tok in children:
            if tok.type == "image":
                ok, host = self._host(tok.attrGet("src") or "", self.image_hosts)
                if not ok:
                    tok = Token("text", "", 0, content=f"[ảnh bị chặn: {host}]")
            elif tok.type == "link_open":
                ok, host = self._host(tok.attrGet("href") or "", self.link_hosts)
                if not ok:
                    dropped.append(host)
                    continue
                dropped.append(None)
                tok.attrSet("rel", "noopener noreferrer nofollow")
            elif tok.type == "link_close":
                host = dropped.pop()
                if host is not None:
                    tok = Token("text", "", 0, content=f" [link bị chặn: {host}]")
            out.append(tok)
        return out

    def render(self, text: str) -> str:
        tokens = self.md.parse(text)
        for tok in tokens:
            if tok.type == "inline" and tok.children:
                tok.children = self._filter(tok.children)
        return self.md.renderer.render(tokens, self.md.options, {})
