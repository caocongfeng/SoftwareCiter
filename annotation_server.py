#!/usr/bin/env python3
"""Local annotation server for real-time save to disk.

Serves annotation HTML and provides a POST endpoint to save
annotation checkpoint JSON directly to disk on every Mark Done.

Usage:
    # Oracle annotations
    python annotation_server.py --dir Gold_Mention_Oracle --port 8765

    # Evaluation annotations
    python annotation_server.py --dir evaluation --port 8766

Then open http://localhost:8765 in browser.
"""

import argparse
import json
import logging
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)


class AnnotationHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves HTML and handles checkpoint save/load."""

    def __init__(self, *args, work_dir: Path = None, checkpoint_file: str = None, **kwargs):
        self.work_dir = work_dir
        self.checkpoint_file = checkpoint_file
        super().__init__(*args, directory=str(work_dir), **kwargs)

    def do_POST(self):
        """Handle POST /api/save — write checkpoint JSON to disk."""
        if self.path == "/api/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                checkpoint = json.loads(body)
                ckpt_path = self.work_dir / self.checkpoint_file

                # Write atomically: write to temp file then rename
                tmp_path = ckpt_path.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, ckpt_path)

                done = checkpoint.get("done_count", "?")
                total = checkpoint.get("total_software", "?")
                logger.info(f"💾 Saved checkpoint: {done}/{total} done → {ckpt_path.name}")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "file": str(ckpt_path)}).encode())

            except Exception as e:
                logger.error(f"Save failed: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        """Handle GET /api/load and static file serving."""
        if self.path == "/api/load":
            ckpt_path = self.work_dir / self.checkpoint_file
            if ckpt_path.exists():
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data.encode())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "not_found"}).encode())
        elif self.path == "/":
            # Serve the EN HTML by default
            html_files = list(self.work_dir.glob("*_annotation_en.html"))
            if html_files:
                self.path = "/" + html_files[0].name
                super().do_GET()
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"No annotation HTML found.")
        else:
            super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress noisy GET logs, keep POST logs."""
        if "POST" in str(args[0]) if args else False:
            logger.info(format % args)


def make_handler(work_dir: Path, checkpoint_file: str):
    """Create handler class with work_dir bound."""
    def handler(*args, **kwargs):
        return AnnotationHandler(*args, work_dir=work_dir, checkpoint_file=checkpoint_file, **kwargs)
    return handler


def main():
    parser = argparse.ArgumentParser(description="Annotation server for real-time save")
    parser.add_argument("--dir", type=str, required=True,
                        help="Working directory (Gold_Mention_Oracle or evaluation)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to serve on (default: 8765)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (auto-detected if not specified)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    work_dir = Path(args.dir).resolve()
    if not work_dir.exists():
        logger.error(f"Directory not found: {work_dir}")
        sys.exit(1)

    # Auto-detect checkpoint file
    if args.checkpoint:
        checkpoint_file = args.checkpoint
    else:
        candidates = list(work_dir.glob("annotation_*_checkpoint.json"))
        if candidates:
            checkpoint_file = candidates[0].name
        else:
            # Default based on dir name
            if "oracle" in work_dir.name.lower():
                checkpoint_file = "annotation_oracle_checkpoint.json"
            else:
                checkpoint_file = "annotation_evaluation_checkpoint.json"

    ckpt_path = work_dir / checkpoint_file
    html_files = list(work_dir.glob("*_annotation_en.html"))

    print(f"\n🚀 Annotation Server")
    print(f"   Directory:  {work_dir}")
    print(f"   Checkpoint: {ckpt_path.name} {'✅ exists' if ckpt_path.exists() else '⚠️ not found'}")
    print(f"   HTML:       {html_files[0].name if html_files else 'not found'}")
    print(f"   URL:        http://localhost:{args.port}")
    print(f"\n   Mark Done will auto-save to: {ckpt_path}")
    print(f"   Press Ctrl+C to stop.\n")

    handler = make_handler(work_dir, checkpoint_file)
    server = HTTPServer(("0.0.0.0", args.port), handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
