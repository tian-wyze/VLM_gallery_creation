"""Tee stdout and stderr to both the terminal and a log file, with timestamps."""

import os
import sys
from datetime import datetime


def _ts():
    return datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ')


class _Tee:
    """File-like object that writes to multiple streams with a timestamp
    prefix at the start of each line.
    """

    def __init__(self, *streams):
        self.streams = streams
        self._at_line_start = True

    def write(self, data):
        if not data:
            return
        # Preserve newline terminators while iterating line-by-line so we can
        # prepend a timestamp at the start of each line.
        for line in data.splitlines(keepends=True):
            if self._at_line_start and line.strip():
                out = _ts() + line
            else:
                out = line
            for s in self.streams:
                try:
                    s.write(out)
                except Exception:
                    pass
            self._at_line_start = line.endswith('\n') or line.endswith('\r')
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        # Preserve terminal-detection behaviour for libraries like tqdm
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def setup_logging(log_path):
    """Redirect stdout/stderr so everything printed also lands in *log_path*.

    The parent directory is created if it doesn't exist. Existing logs are
    appended to, with a banner marking the new run. Returns the open file
    handle so the caller can keep a reference if needed.
    """
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log_file = open(log_path, 'a', buffering=1, encoding='utf-8')

    banner = f"\n{'='*100}\nNew run started at {datetime.now().isoformat(timespec='seconds')}\n{'='*100}\n"
    log_file.write(banner)
    log_file.flush()

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Logging stdout/stderr to {log_path}")
    return log_file
