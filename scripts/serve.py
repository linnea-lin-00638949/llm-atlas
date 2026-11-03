#!/usr/bin/env python3
"""局域网预览：手机与 Mac 连同一个 Wi-Fi 即可访问看板。

用法：python3 scripts/serve.py [--port 8765] [--open]
"""
import argparse
import http.server
import socket
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site"
DOCROOT = SITE if SITE.exists() else ROOT / "dashboard"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCROOT), **kwargs)

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}")


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    ip = lan_ip()
    url = f"http://{ip}:{args.port}/index.html"
    print(f"看板目录：{DOCROOT}")
    print(f"电脑访问：http://127.0.0.1:{args.port}/index.html")
    print(f"手机访问：{url}   （需与 Mac 同一 Wi-Fi）")
    print("按 Ctrl+C 停止。")
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{args.port}/index.html")
    with socketserver.TCPServer(("0.0.0.0", args.port), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
