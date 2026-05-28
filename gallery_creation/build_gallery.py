"""
build_gallery.py
----------------
Two-phase pipeline that builds a family-member gallery from a set of
video clips.

  Phase 1 — DETECT
    Walk every video, sample 1 frame per second, run YOLOv8 person
    detection, save every detected crop to ``state/crops/`` and append
    its metadata (video, frame, timestamp, bbox, conf) to
    ``state/crops_metadata.jsonl``. YOLO is then unloaded so the VLM
    can take the full GPU in phase 2.

  Phase 2 — ASSIGN
    Read ``state/crops_metadata.jsonl`` in order. For each crop, query
    the fine-tuned IDA-VLM with that crop as the query image and the
    current gallery (one representative per identity) as the lettered
    options + a stranger slot. Open a new identity on stranger, append
    to the matched identity otherwise. Writes ``state/decisions.jsonl``
    and ``state/identities.json``.

Splitting like this means:
  * crops are visible / inspectable / curatable before the expensive
    VLM step runs,
  * YOLO doesn't sit in GPU memory while the 7B VLM is loaded, and
  * you can re-run phase 2 in isolation (e.g. with a different
    connector) without redoing detection.

Usage (from IDA-VLM/gallery_creation/):

  python build_gallery.py                       # both phases, default config
  python build_gallery.py --phase detect        # only run YOLO + write crops
  python build_gallery.py --phase assign        # only run VLM assignment
  python build_gallery.py --clean               # wipe state_dir first
  python build_gallery.py --videos example_videos/0a865705a0d0419c89fdf491917bc89e.mp4
"""

import argparse
import gc
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

# Silence noisy library warnings before importing torch / transformers / ultralytics.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("YOLO_VERBOSE", "False")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# IDAVLM lives in IDA-VLM/training/. Add it to sys.path so we can import it.
_HERE = Path(__file__).resolve().parent
_TRAINING_DIR = _HERE.parent / "training"
sys.path.insert(0, str(_TRAINING_DIR))


# ---- Defaults --------------------------------------------------------
# folder = 'example_videos_1'
folder = 'example_videos_2'

DEFAULT_VIDEOS = str(_HERE / folder)
DEFAULT_STATE_DIR = str(_HERE / "state")
DEFAULT_CONNECTOR = (
    "/home/tian.liu/IDA-VLM/training/runs/"
    "20260502_014222_annotated-distractor-sft-qwen7b-WYZEv04_15_token_"
    "Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_"
    "expert_and_image_attn_lr_0.0002_bs_4_captions_False/"
    "ckpts/connector_best_step_38000.pt"
)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_EXPERT_FEATURE = "wyzev0415token"
DEFAULT_INPUT_MODE = "expert_and_image_attn"
DEFAULT_DETECTOR = "wyze"          # "wyze" (YOLOv11, in-house thresholds) | "yolov8" (COCO)
DEFAULT_YOLO = "yolov8n.pt"
# v11_07_25.onnx ships with the vendored WyzeInstanceEmbeddingLib.
DEFAULT_WYZE_DETECTOR_WEIGHTS = str(
    _TRAINING_DIR / "experts" / "wyze_embedding" / "models" / "od" / "v11_07_25.onnx"
)


# ---- Video / detection helpers ---------------------------------------

def list_videos(path):
    """Return a sorted list of video files. Accepts a directory or a single file."""
    p = Path(path)
    if p.is_file():
        return [p]
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


def sample_frames(video_path, fps_target=1.0):
    """Yield (frame_idx, timestamp_s, BGR ndarray) sampled at ~fps_target."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(round(src_fps / fps_target)), 1)
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        yield idx, idx / src_fps, frame
    cap.release()


class PersonDetectorBackend:
    """Common interface for the two person-detector backends.

    Both implementations expose a single ``detect(frame_bgr) -> list[(x1, y1,
    x2, y2, conf)]`` method so the rest of the pipeline doesn't care which
    one is active. Boxes with conf below the per-instance threshold are
    dropped before they reach the caller.
    """

    def detect(self, frame_bgr):
        raise NotImplementedError

    def free(self):
        """Optional: release any GPU memory held by the detector before
        the VLM is loaded in phase 2."""
        pass


class YoloV8PersonDetector(PersonDetectorBackend):
    """Ultralytics YOLOv8 (COCO 80-class) person detector. Used when
    ``--detector yolov8`` is passed."""

    def __init__(self, weights, conf=0.5):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame_bgr):
        results = self.model.predict(
            source=frame_bgr, classes=[0], conf=self.conf, verbose=False,
        )
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                score = float(b.conf[0].cpu().numpy())
                boxes.append((int(x1), int(y1), int(x2), int(y2), score))
        return boxes

    def free(self):
        self.model = None


class WyzePersonDetector(PersonDetectorBackend):
    """Wraps ``WyzeInstanceEmbeddingLib.wyze_embedding.object_detector
    .PersonDetector`` (YOLOv11 trained in-house on Wyze data, with custom
    per-class thresholds). Default backend.

    The Wyze detector ships its own threshold (0.72 for the person
    class); we override it with ``--det_conf`` so the same CLI flag
    controls both backends.
    """

    def __init__(self, weights, conf=0.5, device="cuda"):
        from experts.wyze_embedding.object_detector import PersonDetector
        self.model = PersonDetector(model_path=weights, device=device)
        # PersonDetector hardcodes thresholds[0]=0.72; honor --det_conf
        # so users get one knob across both backends.
        self.model.thresholds[0] = float(conf)

    def detect(self, frame_bgr):
        # detect_persons returns [{'bbox': (x1,y1,x2,y2), 'confidence': c, 'crop': ...}]
        persons = self.model.detect_persons(frame_bgr)
        return [
            (*p["bbox"], float(p["confidence"])) for p in persons
        ]

    def free(self):
        self.model = None


def build_detector(args):
    """Construct the detector backend specified by ``--detector``."""
    if args.detector == "wyze":
        weights = args.wyze_detector_weights
        if not Path(weights).is_file():
            print(f"[detect] Wyze detector weights not found at {weights}.")
            print("[detect] Either download them, point --wyze_detector_weights at "
                  "the correct path, or pass --detector yolov8 to use the "
                  "COCO YOLOv8n fallback.")
            sys.exit(1)
        print(f"[detect] Loading Wyze YOLOv11 person detector: {weights}")
        return WyzePersonDetector(
            weights, conf=args.det_conf,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    elif args.detector == "yolov8":
        print(f"[detect] Loading YOLOv8 weights: {args.yolo_weights}")
        return YoloV8PersonDetector(args.yolo_weights, conf=args.det_conf)
    else:
        raise ValueError(f"unknown --detector: {args.detector}")


def crop_bgr(frame_bgr, bbox, pad=0):
    """Crop with optional padding, clipped to image bounds."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2, _ = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame_bgr[y1:y2, x1:x2]


# ---- Phase 1: detect + crop ------------------------------------------

def run_detect(args, state_dir):
    """Walk all videos, sample frames, detect persons, save crops, and write
    crops_metadata.jsonl (one line per crop, in chronological order)."""
    videos = list_videos(args.videos)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        print(f"No videos found under: {args.videos}")
        return

    print(f"[detect] Will process {len(videos)} video(s):")
    for v in videos:
        print(f"  - {v.name}")
    print(f"[detect] Backend: {args.detector}  (--det_conf={args.det_conf})")
    detector = build_detector(args)

    crops_dir = state_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    meta_path = state_dir / "crops_metadata.jsonl"
    meta_f = open(meta_path, "w")
    crop_counter = 0

    print("\n[detect] Sampling frames + detecting people ...")
    for vi, video in enumerate(videos):
        print(f"\n[detect] [video {vi + 1}/{len(videos)}] {video.name}")
        crops_in_video = 0
        for frame_idx, ts, frame_bgr in sample_frames(video, fps_target=args.fps):
            bboxes = detector.detect(frame_bgr)
            if not bboxes:
                continue
            for bbox in bboxes:
                crop_arr = crop_bgr(frame_bgr, bbox, pad=args.crop_pad)
                if crop_arr.size == 0:
                    continue
                crop_counter += 1
                crop_rel = f"crops/crop_{crop_counter:04d}.jpg"
                cv2.imwrite(str(state_dir / crop_rel), crop_arr)
                meta = {
                    "crop_id": crop_counter,
                    "crop_path": crop_rel,
                    "video": video.name,
                    "frame_idx": frame_idx,
                    "timestamp_s": round(ts, 3),
                    "bbox": [int(b) for b in bbox[:4]],
                    "det_conf": round(bbox[4], 3),
                    "detector": args.detector,
                }
                meta_f.write(json.dumps(meta) + "\n")
                meta_f.flush()
                crops_in_video += 1
        print(f"[detect]   → {crops_in_video} crops from this video")

    meta_f.close()

    # Free the detector before phase 2 loads the 7B VLM.
    detector.free()
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[detect] Done. {crop_counter} crops written to {crops_dir}")
    print(f"[detect] Metadata: {meta_path}")


# ---- Phase 2: gallery state + VLM assignment -------------------------

class GalleryState:
    """Tracks identity → list of crop paths (relative to state_dir).

    The *representative* image used as the gallery slot when querying the
    VLM is the FIRST crop assigned to each identity.
    """

    def __init__(self):
        self.identities = []  # list of {"id": "person_001", "crops": [...]}

    def num_identities(self):
        return len(self.identities)

    def gallery_reps(self):
        return [ident["crops"][0] for ident in self.identities]

    def stranger_letter(self):
        return chr(ord("A") + len(self.identities))

    def open_new_identity(self, crop_path):
        new_id = f"person_{len(self.identities) + 1:03d}"
        self.identities.append({"id": new_id, "crops": [crop_path]})
        return new_id

    def append_to_identity_by_letter(self, letter, crop_path):
        idx = ord(letter) - ord("A")
        if not 0 <= idx < len(self.identities):
            raise ValueError(
                f"letter {letter} out of range for {len(self.identities)} identities"
            )
        self.identities[idx]["crops"].append(crop_path)
        return self.identities[idx]["id"]

    def to_dict(self):
        return {ident["id"]: ident["crops"] for ident in self.identities}


def run_assign(args, state_dir):
    """Read crops_metadata.jsonl, run VLM on each crop, grow gallery online."""
    from inference import IDAVLM  # lazy import — keeps phase 1 fast

    meta_path = state_dir / "crops_metadata.jsonl"
    if not meta_path.exists():
        print(f"[assign] Missing {meta_path}; run --phase detect first.")
        return
    crops_meta = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                crops_meta.append(json.loads(line))
    if not crops_meta:
        print("[assign] crops_metadata.jsonl is empty; nothing to assign.")
        return

    print(f"[assign] Loading IDA-VLM connector: {args.connector_path}")
    vlm = IDAVLM(
        connector_path=args.connector_path,
        model_id=args.model_id,
        feature_mode="expert",
        expert_feature=args.expert_feature,
        input_mode=args.input_mode,
        object_type="person",
    )

    gallery = GalleryState()
    decisions_path = state_dir / "decisions.jsonl"
    decisions_f = open(decisions_path, "w")

    print(f"\n[assign] Assigning {len(crops_meta)} crops ...\n")
    for meta in crops_meta:
        rec = dict(meta)  # start from the detection metadata
        rec.update({
            "gallery_size_before": gallery.num_identities(),
            "gallery_reps_before": list(gallery.gallery_reps()),
            "stranger_letter": None,
            "raw_text": None,
            "answer_letter": None,
            "verdict": None,
            "assigned_to": None,
        })

        if gallery.num_identities() == 0:
            new_id = gallery.open_new_identity(rec["crop_path"])
            rec["verdict"] = "new (gallery_empty)"
            rec["assigned_to"] = new_id
        else:
            query_abs = str(state_dir / rec["crop_path"])
            gallery_abs = [str(state_dir / p) for p in gallery.gallery_reps()]
            rec["stranger_letter"] = gallery.stranger_letter()
            out = vlm.predict(query_abs, gallery_abs)
            rec["raw_text"] = out["raw_text"]
            rec["answer_letter"] = out["answer"]
            if out["answer"] is None:
                new_id = gallery.open_new_identity(rec["crop_path"])
                rec["verdict"] = "new (unparsable)"
                rec["assigned_to"] = new_id
            elif out["answer"] == rec["stranger_letter"]:
                new_id = gallery.open_new_identity(rec["crop_path"])
                rec["verdict"] = "new (stranger)"
                rec["assigned_to"] = new_id
            else:
                try:
                    matched_id = gallery.append_to_identity_by_letter(
                        out["answer"], rec["crop_path"],
                    )
                    rec["verdict"] = "matched"
                    rec["assigned_to"] = matched_id
                except ValueError as e:
                    new_id = gallery.open_new_identity(rec["crop_path"])
                    rec["verdict"] = f"new (out_of_range: {e})"
                    rec["assigned_to"] = new_id

        decisions_f.write(json.dumps(rec) + "\n")
        decisions_f.flush()
        print(
            f"  crop {rec['crop_id']:04d} "
            f"(t={rec['timestamp_s']:5.2f}s, conf={rec['det_conf']:.2f}, "
            f"gallery_before={rec['gallery_size_before']}) → "
            f"{rec['verdict']} → {rec['assigned_to']}"
            + (f"  [model said {rec['answer_letter']}]" if rec["answer_letter"] else "")
        )

    decisions_f.close()

    with open(state_dir / "identities.json", "w") as f:
        json.dump(gallery.to_dict(), f, indent=2)

    print("\n" + "=" * 60)
    print(f"[assign] {gallery.num_identities()} identities discovered "
          f"across {len(crops_meta)} crops.")
    for ident in gallery.identities:
        print(f"  {ident['id']}: {len(ident['crops'])} crops")
    print(f"\n[assign] State written to: {state_dir}")
    print(f"  identities: {state_dir / 'identities.json'}")
    print(f"  decisions:  {decisions_path}")
    print(
        f"\nNext: python visualize.py --state_dir {state_dir} "
        f"--out {state_dir / 'visualization.html'}"
    )


# ---- Main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Two-phase pipeline: YOLO person detection → IDA-VLM gallery "
            "assignment. See module docstring for outputs."
        ),
    )
    parser.add_argument("--phase", choices=["detect", "assign", "all"],
                        default="all",
                        help="Which phase(s) to run. Default: all "
                             "(detect then assign in one process, with "
                             "YOLO freed between).")
    parser.add_argument("--videos", default=DEFAULT_VIDEOS,
                        help="Directory of videos OR a single video path.")
    parser.add_argument("--state_dir", default=DEFAULT_STATE_DIR,
                        help="Where to write crops/, metadata, "
                             "decisions, identities.json.")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames sampled per second of source video.")
    parser.add_argument("--det_conf", type=float, default=0.5,
                        help="YOLO confidence threshold (class=person). "
                             "Boxes below this score are dropped before "
                             "any crop is saved.")
    parser.add_argument("--crop_pad", type=int, default=8,
                        help="Pixel padding on each side of YOLO bbox.")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Cap on number of videos processed (debug).")
    parser.add_argument("--detector", choices=["wyze", "yolov8"],
                        default=DEFAULT_DETECTOR,
                        help="Which person detector to use. "
                             "'wyze' (default): the YOLOv11 PersonDetector "
                             "from WyzeInstanceEmbeddingLib, trained "
                             "in-house on Wyze footage. "
                             "'yolov8': the off-the-shelf YOLOv8n COCO "
                             "checkpoint via ultralytics (fallback).")
    parser.add_argument("--wyze_detector_weights",
                        default=DEFAULT_WYZE_DETECTOR_WEIGHTS,
                        help="Path to the Wyze YOLOv11 ONNX/PT weights. "
                             "Only used when --detector wyze.")
    parser.add_argument("--yolo_weights", default=DEFAULT_YOLO,
                        help="YOLOv8 weights filename or path "
                             "(auto-downloaded by ultralytics if missing). "
                             "Only used when --detector yolov8.")
    parser.add_argument("--connector_path", default=DEFAULT_CONNECTOR)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--expert_feature", default=DEFAULT_EXPERT_FEATURE)
    parser.add_argument("--input_mode", default=DEFAULT_INPUT_MODE)
    parser.add_argument("--clean", action="store_true",
                        help="Wipe state_dir before starting.")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    if args.clean and state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("detect", "all"):
        run_detect(args, state_dir)
    if args.phase in ("assign", "all"):
        run_assign(args, state_dir)


if __name__ == "__main__":
    main()
