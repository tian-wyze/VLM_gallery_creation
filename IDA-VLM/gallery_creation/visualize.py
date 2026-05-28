"""
visualize.py
------------
Read the state directory produced by ``build_gallery.py`` and emit a
single self-contained HTML file that visualizes:

  1. The final family-member gallery — one row per identity, all crops
     for that identity laid out left-to-right, with the "rep" badge on
     the first (the gallery slot used at query time).
  2. The decision log — one card per crop in chronological order,
     showing the query crop, the gallery state at that moment (per-letter
     thumbnails + stranger slot), the model's raw response, and the
     verdict (matched / new).

The HTML references image paths relative to the state_dir, so as long as
you open the HTML from inside state_dir (or pass --copy_images) the
thumbnails will load.

Usage (from IDA-VLM/gallery_creation/):

  python visualize.py
  python visualize.py --state_dir state --out state/visualization.html
"""

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_STATE_DIR = str(_HERE / "state")
DEFAULT_OUT = str(_HERE / "state" / "visualization.html")


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 20px; background: #f5f6f8; color: #222; }
h1 { margin-top: 0; }
h2 { border-bottom: 2px solid #ddd; padding-bottom: 8px; margin-top: 32px; }
.identity-row { display: flex; align-items: center; gap: 12px;
                background: #fff; padding: 12px; border-radius: 8px;
                margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.identity-label { font-weight: 600; min-width: 110px; }
.identity-count { color: #888; font-size: 0.85em; margin-left: 6px; }
.crop-strip { display: flex; gap: 8px; overflow-x: auto; padding: 4px 0; }
.crop-thumb { position: relative; flex: 0 0 auto; }
.crop-thumb img { height: 140px; display: block; border-radius: 4px;
                  border: 2px solid transparent; }
.crop-thumb.rep img { border-color: #2da44e; }
.crop-thumb .badge { position: absolute; top: 2px; left: 2px;
                     background: #2da44e; color: white;
                     font-size: 0.7em; padding: 2px 4px; border-radius: 3px; }
.crop-thumb .caption { font-size: 0.72em; color: #555; margin-top: 2px;
                       max-width: 110px; word-break: break-all; }

.decision-card { background: #fff; padding: 14px; border-radius: 8px;
                 margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.decision-header { display: flex; justify-content: space-between;
                   align-items: baseline; margin-bottom: 8px; }
.decision-id { font-weight: 700; font-size: 1.05em; }
.decision-meta { font-family: ui-monospace, "SF Mono", Menlo, monospace;
                 font-size: 0.82em; color: #666; }
.decision-body { display: flex; gap: 18px; align-items: flex-start;
                 flex-wrap: wrap; }
.column { display: flex; flex-direction: column; gap: 6px; }
.column .col-label { font-size: 0.78em; font-weight: 600; color: #555;
                     text-transform: uppercase; letter-spacing: 0.04em; }
.query img { height: 160px; border-radius: 4px;
             border: 2px solid #0969da; }
.gallery-thumbs { display: flex; gap: 6px; }
.gallery-thumbs .slot { text-align: center; }
.gallery-thumbs .slot img { height: 140px; border-radius: 4px;
                            border: 2px solid #ddd; }
.gallery-thumbs .letter { font-weight: 700; font-size: 0.95em;
                          color: #333; margin-bottom: 2px; }
.gallery-thumbs .stranger { width: 88px; height: 140px;
                            border: 2px dashed #aaa; border-radius: 4px;
                            display: flex; align-items: center;
                            justify-content: center; text-align: center;
                            font-size: 0.78em; color: #666; padding: 4px; }
.outcome { min-width: 220px; }
.outcome .raw-text { background: #f1f3f5; padding: 8px 10px;
                     border-radius: 4px; font-family: ui-monospace, Menlo, monospace;
                     font-size: 0.85em; white-space: pre-wrap; max-width: 480px; }
.verdict { padding: 4px 8px; border-radius: 4px; font-weight: 600;
           display: inline-block; }
.verdict.matched   { background: #ddf4e4; color: #1a7f37; }
.verdict.new       { background: #fff8c5; color: #9a6700; }
.verdict.unparsable{ background: #ffebe9; color: #cf222e; }

.summary { background: #fff; padding: 14px 16px; border-radius: 8px;
           margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.summary table { border-collapse: collapse; }
.summary td { padding: 3px 12px 3px 0; font-size: 0.92em; }
.summary td:first-child { color: #666; }
"""


def _data_uri(abs_path):
    """Read an image and return a base64 data: URI (or '' on missing file)."""
    p = Path(abs_path)
    if not p.is_file():
        return ""
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None:
        mime = "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _make_img_tag_fn(state_dir, embed):
    """Return a function `img_tag(rel_path, alt)` that renders <img>.

    If `embed=True`, the image bytes are base64-encoded into the src so the
    HTML is fully self-contained (works in VS Code preview and in any
    browser without needing file:// access). If False, the src is the
    relative path and the HTML must be opened in a real browser from
    inside `state_dir` for the thumbnails to load.

    Non-embedded mode adds ``loading="lazy"`` so off-screen thumbnails
    are only fetched when scrolled into view — big win when the page is
    served over a slow SSH tunnel with dozens of crop refs in the
    decision log.
    """
    def img_tag(rel_path, alt=""):
        if not rel_path:
            return ""
        if embed:
            src = _data_uri(state_dir / rel_path)
            lazy = ""  # base64 data URIs have nothing to lazy-load
        else:
            src = html.escape(rel_path)
            lazy = ' loading="lazy" decoding="async"'
        return f'<img src="{src}" alt="{html.escape(alt)}"{lazy}>'
    return img_tag


def _verdict_class(verdict):
    if verdict is None:
        return "unparsable"
    if verdict.startswith("matched"):
        return "matched"
    if "unparsable" in verdict:
        return "unparsable"
    return "new"


def render_identity_section(identities, img_tag):
    parts = ['<h2>Final family gallery</h2>']
    if not identities:
        parts.append('<p>No identities discovered.</p>')
        return "\n".join(parts)
    for ident_id, crops in identities.items():
        thumbs = []
        for i, crop_path in enumerate(crops):
            cls = "crop-thumb rep" if i == 0 else "crop-thumb"
            badge = '<span class="badge">rep</span>' if i == 0 else ''
            thumbs.append(
                f'<div class="{cls}">{badge}{img_tag(crop_path, ident_id)}'
                f'<div class="caption">{html.escape(crop_path)}</div></div>'
            )
        parts.append(
            f'<div class="identity-row">'
            f'<div class="identity-label">{html.escape(ident_id)}'
            f'<span class="identity-count">({len(crops)} crops)</span></div>'
            f'<div class="crop-strip">{"".join(thumbs)}</div>'
            f'</div>'
        )
    return "\n".join(parts)


def render_decision_section(decisions, img_tag):
    parts = ['<h2>Decision log</h2>']
    if not decisions:
        parts.append('<p>No decisions recorded.</p>')
        return "\n".join(parts)

    for rec in decisions:
        verdict = rec.get("verdict") or "unparsable"
        vclass = _verdict_class(verdict)

        # Gallery state at the moment of decision: letters A.. for reps,
        # then a dashed "stranger" slot at the end.
        gallery_html = []
        for i, rep in enumerate(rec.get("gallery_reps_before", [])):
            letter = chr(ord("A") + i)
            gallery_html.append(
                f'<div class="slot">'
                f'<div class="letter">{letter}</div>'
                f'{img_tag(rep)}'
                f'</div>'
            )
        stranger_letter = rec.get("stranger_letter")
        if stranger_letter:
            gallery_html.append(
                f'<div class="slot">'
                f'<div class="letter">{html.escape(stranger_letter)}</div>'
                f'<div class="stranger">(stranger /<br/>not in gallery)</div>'
                f'</div>'
            )
        elif rec["gallery_size_before"] == 0:
            gallery_html.append(
                '<div class="slot"><div class="letter">—</div>'
                '<div class="stranger">(gallery<br/>was empty)</div></div>'
            )

        # Header line: crop id, source video, frame, timestamp, det_conf
        meta = (
            f'video={html.escape(rec["video"])}  '
            f'frame={rec["frame_idx"]}  '
            f't={rec["timestamp_s"]:.2f}s  '
            f'conf={rec["det_conf"]:.2f}'
        )

        # Verdict block
        raw = html.escape(rec.get("raw_text") or "(no VLM call — gallery was empty)")
        answer_letter = rec.get("answer_letter")
        answer_disp = (
            f"<b>Model answer:</b> {html.escape(str(answer_letter))}<br/>"
            if answer_letter is not None else ""
        )

        parts.append(f'''
<div class="decision-card">
  <div class="decision-header">
    <div class="decision-id">crop_{rec["crop_id"]:04d}</div>
    <div class="decision-meta">{meta}</div>
  </div>
  <div class="decision-body">
    <div class="column query">
      <div class="col-label">Query</div>
      {img_tag(rec["crop_path"])}
    </div>
    <div class="column">
      <div class="col-label">Gallery state at query time</div>
      <div class="gallery-thumbs">{"".join(gallery_html)}</div>
    </div>
    <div class="column outcome">
      <div class="col-label">Outcome</div>
      <div class="raw-text">{raw}</div>
      <div>{answer_disp}<b>Assigned to:</b> {html.escape(str(rec.get("assigned_to")))}</div>
      <div><span class="verdict {vclass}">{html.escape(verdict)}</span></div>
    </div>
  </div>
</div>''')
    return "\n".join(parts)


def render_summary(decisions, identities):
    total = len(decisions)
    new = sum(1 for r in decisions if (r.get("verdict") or "").startswith("new"))
    matched = sum(1 for r in decisions if (r.get("verdict") or "") == "matched")
    return f'''
<div class="summary">
  <table>
    <tr><td>Total crops</td><td>{total}</td></tr>
    <tr><td>Identities discovered</td><td>{len(identities)}</td></tr>
    <tr><td>Crops matched to existing identity</td><td>{matched}</td></tr>
    <tr><td>Crops opened a new identity</td><td>{new}</td></tr>
  </table>
</div>'''


def load_state(state_dir):
    """Read identities.json + decisions.jsonl from state_dir.

    Returns (identities_dict, decisions_list). Missing files are returned
    as empty containers so the renderer can still produce a coherent
    "no data yet" page during a partial pipeline run.
    """
    state_dir = Path(state_dir)
    identities_path = state_dir / "identities.json"
    decisions_path = state_dir / "decisions.jsonl"

    identities = {}
    if identities_path.exists():
        with open(identities_path) as f:
            identities = json.load(f)

    decisions = []
    if decisions_path.exists():
        with open(decisions_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    decisions.append(json.loads(line))
    return identities, decisions


def render_html(state_dir, embed=True):
    """Return the full HTML document as a string, reading the current
    contents of state_dir/identities.json and state_dir/decisions.jsonl.

    embed=True: thumbnails are base64-embedded data URIs (works in any
                viewer including VS Code preview, but bigger file).
    embed=False: thumbnails are relative paths under state_dir (smaller,
                requires an HTTP server or a real browser to load).
    """
    state_dir = Path(state_dir)
    identities, decisions = load_state(state_dir)
    img_tag = _make_img_tag_fn(state_dir, embed=embed)

    body = []
    body.append("<h1>Family-Member Gallery — Build Trace</h1>")
    body.append(f'<p>State directory: <code>{html.escape(str(state_dir))}</code></p>')
    body.append(render_summary(decisions, identities))
    body.append(render_identity_section(identities, img_tag))
    body.append(render_decision_section(decisions, img_tag))

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Family-Member Gallery — Build Trace</title>
<style>{CSS}</style>
</head>
<body>
{"".join(body)}
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output HTML path.")
    parser.add_argument("--no_embed", action="store_true",
                        help="Use relative image paths instead of "
                             "base64-embedded data URIs. The resulting "
                             "HTML is smaller but only renders correctly "
                             "when opened in a real browser from inside "
                             "state_dir; embedded mode (the default) "
                             "works in VS Code preview too.")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_doc = render_html(state_dir, embed=not args.no_embed)
    with open(out_path, "w") as f:
        f.write(html_doc)

    identities, decisions = load_state(state_dir)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote: {out_path}  ({size_mb:.2f} MB)")
    print(f"  identities: {len(identities)}")
    print(f"  decisions:  {len(decisions)}")
    print(f"  mode:       {'embedded (self-contained)' if not args.no_embed else 'relative paths'}")
    print(f"Open it with: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
