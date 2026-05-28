"""
serve.py
--------
Tiny stdlib HTTP server that exposes the family-gallery viz over the
network. Re-renders the HTML on every request, so refreshing the page
picks up the latest pipeline state without you re-running visualize.py.

Typical remote-machine workflow:

  # On the remote machine where the state/ dir lives:
  cd IDA-VLM/gallery_creation
  python serve.py                # default port 8088

  # On your local laptop:
  ssh -L 8088:localhost:8088 <user>@<remote-host>

  # Then open in Chrome (on the laptop):
  http://localhost:8088/

Everything (HTML, JPGs, JSON, JSONL) is served from --state_dir. The
server binds to 127.0.0.1 by default so it's only reachable through the
SSH tunnel — pass --host 0.0.0.0 to expose it on the LAN (use with
caution; no auth).
"""

import argparse
import errno
import os
import socket
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReusableHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer (one request → one thread, so the browser's
    6 parallel image fetches actually run in parallel) plus SO_REUSEADDR
    so restarting right after a Ctrl-C works without waiting out the
    kernel's TIME_WAIT window (~60 s on Linux).

    Without threading the stdlib HTTPServer is strictly serial, which
    turns a page with ~100 image refs into a slow drip — the dominant
    cost of loading the viz, not HTML rendering.
    """
    allow_reuse_address = True
    daemon_threads = True  # don't block server shutdown on in-flight requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from visualize import render_html  # noqa: E402

DEFAULT_STATE_DIR = str(_HERE / "state")


class GalleryHandler(SimpleHTTPRequestHandler):
    """Serves images / JSON straight from disk, and re-renders the HTML
    index page on each request to ``/`` or ``/index.html``."""

    state_dir = None  # set by main() before serving

    def log_message(self, fmt, *args):
        # Quieter than the default — only print "method path status".
        sys.stderr.write(
            f"[{self.log_date_time_string()}] "
            f"{self.command} {self.path} → {args[1] if len(args) > 1 else '?'}\n"
        )

    def end_headers(self):
        # Browser caching policy:
        # - The rendered HTML at / must NOT be cached, so refresh always
        #   reflects the latest state files.
        # - Crop images, JSON, and JSONL DO get cached (max-age=1h) so
        #   the browser doesn't re-fetch them every refresh — critical
        #   over SSH tunnels where each crop is ~100 KB.
        # Crop filenames are append-only and monotonic, so cached bytes
        # are always still valid for that crop_id.
        if self.path in ("/", "/index.html"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html_doc = render_html(self.state_dir, embed=False)
            payload = html_doc.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        # Everything else (images, JSON, JSONL) → static file under state_dir.
        # Parent class respects the `directory=` ctor arg we set in main().
        return super().do_GET()


def _hostname():
    try:
        return socket.gethostname()
    except Exception:
        return "<remote-host>"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Serve the family-member gallery viz over HTTP. Browse to / "
            "for the rendered HTML; static files (crops, jsonl, json) "
            "are served at their relative paths under state_dir."
        ),
    )
    parser.add_argument("--state_dir", default=DEFAULT_STATE_DIR,
                        help="State directory written by build_gallery.py.")
    parser.add_argument("--port", type=int, default=8088,
                        help="Port to listen on. Default 8088 "
                             "(8080 is commonly used by Jupyter). "
                             "If this port is taken, the server scans "
                             "upward for a free one — see "
                             "--no_auto_fallback / --port_scan_limit.")
    parser.add_argument("--no_auto_fallback", dest="auto_fallback",
                        action="store_false",
                        help="Fail immediately if --port is taken "
                             "instead of scanning upward for a free "
                             "port.")
    parser.add_argument("--port_scan_limit", type=int, default=20,
                        help="When the requested port is busy, try up "
                             "to this many subsequent ports before "
                             "giving up.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind to. Default 127.0.0.1 "
                             "(loopback only — safe behind SSH port "
                             "forwarding). Pass 0.0.0.0 to expose on "
                             "the LAN — there is no auth, use with care.")
    args = parser.parse_args()

    state_dir = Path(args.state_dir).resolve()
    if not state_dir.is_dir():
        print(f"state_dir does not exist: {state_dir}")
        print("Run `python build_gallery.py` first.")
        sys.exit(1)

    GalleryHandler.state_dir = state_dir
    handler_cls = partial(GalleryHandler, directory=str(state_dir))

    # Try the requested port first; on EADDRINUSE, scan upward up to
    # `port_scan_limit` candidates and pick the first free one. This
    # avoids the "guess another port" loop on shared hosts where common
    # ports (8080, 8088, 8888, ...) are routinely claimed by Jupyter /
    # other users' http.server processes.
    server = None
    chosen_port = None
    candidates = [args.port] + (
        [args.port + i for i in range(1, args.port_scan_limit + 1)]
        if args.auto_fallback else []
    )
    for p in candidates:
        try:
            server = ReusableHTTPServer((args.host, p), handler_cls)
            chosen_port = p
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            if p == args.port:
                print(f"Port {args.port} is in use on {args.host}; "
                      f"scanning {args.port + 1}..{args.port + args.port_scan_limit} ...")
    if server is None:
        print(f"All candidate ports {args.port}..{args.port + args.port_scan_limit} "
              f"are taken. Pick one explicitly with --port, or:")
        print(f"  ss -ltnp | grep ':{args.port}'")
        print(f"  lsof -iTCP:{args.port} -sTCP:LISTEN")
        sys.exit(1)
    if chosen_port != args.port:
        print(f"→ Bound to free port {chosen_port} "
              f"(requested {args.port} was in use).\n")

    user = os.environ.get("USER", "user")
    host = _hostname()
    print(f"Serving {state_dir} on http://{args.host}:{chosen_port}/\n")
    print("From your local laptop, run:")
    print(f"  ssh -L {chosen_port}:localhost:{chosen_port} {user}@{host}")
    print("then open in Chrome on the laptop:")
    print(f"  http://localhost:{chosen_port}/")
    print("\nCtrl-C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
