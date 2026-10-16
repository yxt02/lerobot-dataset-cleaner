#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_lerobot_dataset.py — LeRobot v2.1 数据清洗脚本
====================================================

策略（按优先级）：
  1) 启停冗余帧  轨迹开头/结尾持续静止  → 裁剪无效等待区间
  2) 迟疑帧      长时间无有效操作/无状态变化 → 标记后选择性剔除（默认只标记）
  3) 模糊帧      Laplacian 方差 / 图像清晰度   → 标记异常片段（只标记）

原则：优先清理无效片段，不直接删除所有被判定为异常的帧。
      只有「启停冗余」默认裁剪；迟疑/模糊默认写入标记，由 --trim-hesitation 等开关决定是否剔除。

检测信号说明（重要）：
  - 移动底座的机器人在行驶时手臂关节不动、AGV 通道可能全 0，
    仅用本体关节速度会把「行驶段」误判为静止段。
  - 因此优先用「视频背景运动 ∪ 本体运动」作为运动信号；
    视频损坏/不可读时退化为本体信号，此时边界裁剪上限为 --max-boundary-trim-sec，防止误删。
  - 视频即使缺 moov（未 finalize），也可按 mdat 内 AVCC 流抢救解码前缀帧用于检测。

用法：
  # 1) 只出报告（推荐先跑这个）
  python3 -I clean_lerobot_dataset.py --src <数据集> --report

  # 2) 生成清洗后的数据集（裁启停冗余；迟疑/模糊只标记）
  python3 -I clean_lerobot_dataset.py --src <数据集> --dst <输出> --apply

  # 3) 迟疑帧也剔除（选择性剔除）
  python3 -I clean_lerobot_dataset.py --src <数据集> --dst <输出> --apply --trim-hesitation

输出：
  <dst>/                 清洗后的 LeRobot 数据集
  <dst>/meta/cleaning_report.json   全部检测结果
  <dst>/meta/phases.jsonl           每集启/停/迟疑/模糊标记（最终坐标 + 原始坐标）
  <dst>/video_manifest.json         视频需同步裁剪的帧数（视频修复后用它对齐）
"""
import argparse, json, os, struct, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

FPS_DEFAULT = 30.0
JOINT_MOTION_KEYS = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow_yaw", "left_elbow_pitch",
    "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    "lower_lifting", "upper_lifting", "waist_pitch", "waist_yaw",
    "head_yaw", "head_pitch",
]
AGV_KEYS = ["agv_linear_x", "agv_linear_y", "agv_angular_z"]


# ----------------------------------------------------------------- utilities
# motion detection is robot-agnostic: use every proprio dimension available.
# JOINT_MOTION_KEYS / AGV_KEYS are only used to *prefer* named dims when present
# (e.g. to know whether AGV/base velocity channels are alive on a z0 dataset).
JOINT_MOTION_KEYS = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow_yaw", "left_elbow_pitch",
    "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    "lower_lifting", "upper_lifting", "waist_pitch", "waist_yaw",
    "head_yaw", "head_pitch",
]
AGV_KEYS = ["agv_linear_x", "agv_linear_y", "agv_angular_z"]


def to_arr(x):
    if isinstance(x, np.ndarray):
        return np.array([to_arr(e) for e in x]) if x.dtype == object else x.astype(np.float64)
    if isinstance(x, (list, tuple)):
        return np.array([to_arr(e) for e in x], dtype=np.float64)
    return np.float64(x)


def stack_col(col):
    return np.array([to_arr(v) for v in col])


def smooth(x, w):
    if w <= 1 or len(x) <= w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def runs_of(mask):
    """Return list of (start, end_exclusive) for True runs in a boolean mask."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


# ----------------------------------------------------------------- video IO
def find_mdat_payload(data):
    off = 0
    while off + 8 <= len(data):
        size, typ = struct.unpack(">I4s", data[off:off + 8])
        if typ == b"mdat":
            return data[off + 8:]
        if size < 8:
            return None
        if size == 1:
            size = struct.unpack(">Q", data[off + 8:off + 16])[0]
        elif size == 0:
            size = len(data) - off
        off += size
    return None


def avcc_to_annexb(payload):
    out = bytearray()
    p = 0
    while p + 4 <= len(payload):
        ln = struct.unpack(">I", payload[p:p + 4])[0]
        if ln == 0 or ln > len(payload) - p - 4 or ln > 5_000_000:
            break
        out += b"\x00\x00\x00\x01" + payload[p + 4:p + 4 + ln]
        p += 4 + ln
    return bytes(out)


def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def decode_video_gray(mp4, max_frames=2000, width=160, height=90):
    """Decode up to max_frames gray frames. Handles truncated mp4 (no moov)
    by salvaging the mdat/AVCC prefix. Returns (frames uint8[N,H,W], note) or (None, reason)."""
    FF = get_ffmpeg()
    if not FF:
        return None, "no ffmpeg available"

    def run_ffmpeg(src, is_raw):
        cmd = [FF, "-v", "error", "-i", str(src), "-frames:v", str(max_frames),
               "-vf", f"scale={width}:{height}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
        r = subprocess.run(cmd, capture_output=True)
        fsz = width * height
        n = len(r.stdout) // fsz
        if n < 2:
            return None
        return np.frombuffer(r.stdout[:n * fsz], dtype=np.uint8).reshape(n, height, width)

    data = Path(mp4).read_bytes()
    has_moov = b"moov" in data

    # 1) normal path: intact mp4 (moov present)
    if has_moov:
        frames = run_ffmpeg(mp4, False)
        if frames is not None:
            return frames, "ok"
        return None, "decode_failed"

    # 2) salvage path: truncated mp4, decode the mdat/AVCC prefix
    payload = find_mdat_payload(data)
    if payload is None:
        return None, "no mdat payload"
    raw = (payload if (payload[:4] != b"\x00\x00\x00\x01" and len(payload) > 4)
           else payload)
    # accept both AVCC (4-byte length prefixes) and annex-B
    if raw[:4] == b"\x00\x00\x00\x01" or raw[:3] == b"\x00\x00\x01":
        blob = raw
    else:
        blob = avcc_to_annexb(raw)
    td = tempfile.mkdtemp(prefix="lrbqc_")
    h264 = os.path.join(td, "s.h264")
    with open(h264, "wb") as f:
        f.write(blob)
    frames = run_ffmpeg(h264, True)
    if frames is None:
        return None, "decode_failed"
    return frames, "truncated_mp4(salvaged_prefix)"


def laplacian_var(frames):
    """Per-frame Laplacian variance (blur score; higher = sharper)."""
    # kernel: [[0,1,0],[1,-4,1],[0,1,0]]
    f = frames.astype(np.float32)
    lap = (f[:, 1:-1, 2:] + f[:, 1:-1, :-2] + f[:, 2:, 1:-1] + f[:, :-2, 1:-1]
           - 4 * f[:, 1:-1, 1:-1])
    return lap.reshape(len(f), -1).var(axis=1)


def bg_motion(frames):
    """Background (upper strip) mean-abs-diff motion energy; smoothed."""
    bg = frames[:, : max(4, frames.shape[1] // 3), :]
    m = np.abs(np.diff(bg.astype(np.float32), axis=0)).mean(axis=(1, 2))
    return smooth(m, 9)


# ----------------------------------------------------------------- detection
def detect_episode(df, vid_frames, cfg):
    """Return detection dict for one episode (all indices in source coords).

    Motion signal is robot-agnostic: per-frame max |Δ| across EVERY proprio
    column of kind ``observation.*_positions`` (joint + finger + ...), so the
    script works on z0 (25-DoF), z02 (22-DoF) and other layouts alike.
    """
    n = len(df)
    names = cfg["joint_names"]

    # collect every proprio position feature present in this parquet
    prop_cols = [c for c in df.columns
                 if c.startswith("observation.") and c.endswith("_positions")
                 and df[c].dtype == object]
    if not prop_cols:
        # fall back to the declared joint feature even if stored as plain columns
        prop_cols = [c for c in ("observation.joint_positions",) if c in df.columns]
    mats = []
    for c in prop_cols:
        try:
            mats.append(np.stack(df[c].to_numpy()).astype(np.float64))
        except Exception:
            continue
    o_all = np.concatenate(mats, axis=1) if mats else np.zeros((n, 1))

    # named dims (optional diagnostics)
    jidx = [names.index(k) for k in JOINT_MOTION_KEYS if k in names]
    aidx = [names.index(k) for k in AGV_KEYS if k in names]

    # per-dim normalization so one threshold works across units
    # (radians vs raw encoder counts on some hands)
    scale = o_all.max(0) - o_all.min(0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    joint_d = smooth(np.abs(np.diff(o_all, axis=0) / scale).max(1), 7)
    if aidx:
        oj = np.stack(df["observation.joint_positions"].to_numpy()).astype(np.float64)
        agv = oj[:, aidx]
        agv_d = smooth(np.abs(np.diff(agv, axis=0)).max(1), 7)
    else:
        agv_d = np.zeros(n - 1)
    agv_used = bool(np.any(oj[:, aidx].std(0) > 1e-6)) if aidx else False

    jthr = cfg["joint_thr"]
    arm_active = joint_d > jthr
    agv_active = agv_d > cfg["agv_thr"]
    state_active = arm_active | agv_active

    video_info = None
    bg = None
    bg_active = None
    blur = None
    if vid_frames is not None:
        bg = bg_motion(vid_frames)
        bg_active = bg > cfg["bg_thr"]
        blur = laplacian_var(vid_frames)
        video_info = dict(decoded=int(len(vid_frames)), bg_mean=float(bg.mean()))

    # combined motion signal (pad video to n-1)
    def fit_to(x, m):
        y = np.zeros(n - 1, dtype=bool)
        L = min(len(m), n - 1)
        y[:L] = m[:L]
        return y

    if bg_active is not None:
        combined = state_active | fit_to(None, bg_active)
    else:
        combined = state_active

    # ---- boundaries: leading / trailing static (only where NOTHING moves)
    idle = ~combined
    lead = 0
    while lead < len(idle) and idle[lead]:
        lead += 1
    trail = 0
    while trail < len(idle) and idle[len(idle) - 1 - trail]:
        trail += 1

    # safety cap when we cannot see base motion (broken video + dead AGV)
    can_see_base = (bg_active is not None) or agv_used
    cap = cfg["max_boundary_trim"] if not can_see_base else None
    lead_trim = lead if cap is None else min(lead, cap)
    trail_trim = trail if cap is None else min(trail, cap)

    # ---- interior idle gaps (candidate 迟疑帧 segments)
    gaps = []
    idx = np.where(state_active)[0]
    if len(idx) > 1:
        for a, b in zip(idx, idx[1:]):
            if b - a > 1:
                gaps.append((int(a) + 1, int(b)))  # idle frames [a, b)
    min_gap = int(cfg["min_idle_sec"] * cfg["fps"])
    gaps = [g for g in gaps if g[1] - g[0] >= min_gap]

    # classify each gap: travel (base moving → keep) vs hesitation (nothing moving)
    classified = []
    travel_gap = None
    scored = []
    for a, b in gaps:
        if bg is not None:
            seg = bg[a:min(b, len(bg))]
            bg_hi = float(seg.mean()) > cfg["bg_thr"] if len(seg) else False
            kind = "travel_hold" if bg_hi else "hesitation"
            score = float(seg.mean()) if len(seg) else 0.0
        else:
            score = float(b - a)
            kind = None  # decided below
        scored.append((a, b, score, kind))
    if bg is None and scored:
        # no video: keep the single longest gap as travel/hold, rest = hesitation
        longest = max(scored, key=lambda t: t[2])
        travel_gap = (longest[0], longest[1])
        for a, b, s, k in scored:
            k = "travel_hold" if (a, b) == travel_gap else "hesitation"
            classified.append(dict(start=a, end=b, kind=k, seconds=(b - a) / cfg["fps"],
                                   score=s))
    else:
        for a, b, s, k in scored:
            classified.append(dict(start=a, end=b, kind=k, seconds=(b - a) / cfg["fps"],
                                   score=s))

    # arm action onset (迟疑帧 as the base→arm switch marker)
    hes = None
    run = 0
    for i, v in enumerate(joint_d):
        run = run + 1 if v > jthr else 0
        if run >= 15:
            hes = i - 14
            break
    arm_end = int(idx[-1]) if len(idx) else n - 1

    # ---- blur segments (mark only)
    blur_segments = []
    if blur is not None:
        bad = blur < cfg["blur_thr"]
        for a, b in runs_of(bad):
            if b - a >= int(cfg["min_blur_sec"] * cfg["fps"]):
                blur_segments.append(dict(start=int(a), end=int(b),
                                          seconds=(b - a) / cfg["fps"],
                                          mean_score=float(blur[a:b].mean())))

    return dict(
        length=n,
        video=video_info,
        agv_channel_used=agv_used,
        can_see_base_motion=can_see_base,
        boundaries=dict(leading_static=lead, trailing_static=trail,
                        leading_trim=lead_trim, trailing_trim=trail_trim),
        hesitation_frame=hes,
        arm_end_frame=arm_end,
        idle_gaps=classified,
        blur_segments=blur_segments,
    )


# ----------------------------------------------------------------- rewriting
def recompute_stats(df, state_keys):
    stats = {}
    for k in state_keys:
        if k not in df.columns:
            continue
        arr = stack_col(df[k].to_numpy()) if df[k].dtype == object else df[k].to_numpy().astype(np.float64)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            stats[k] = dict(min=[float(arr.min())], max=[float(arr.max())],
                            mean=[float(arr.mean())], std=[float(arr.std())],
                            count=[int(len(arr))])
        else:
            axis = tuple(range(1, arr.ndim))
            stats[k] = dict(
                min=np.asarray(arr.min(axis=axis), np.float32).tolist(),
                max=np.asarray(arr.max(axis=axis), np.float32).tolist(),
                mean=np.asarray(arr.mean(axis=axis), np.float32).tolist(),
                std=np.asarray(arr.std(axis=axis), np.float32).tolist(),
                count=[int(len(arr))])
    return stats


def write_dataset(src, dst, info, episodes, detections, cfg, apply_trim):
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    state_keys = [k for k in info["features"] if k not in video_keys]

    new_eps, new_stats, phase_rows = [], [], []
    video_manifest = {"note": "apply these frame cuts to videos AFTER repair", "episodes": {}}
    gidx = 0
    total = 0

    for e in episodes:
        ei = e["episode_index"]
        det = detections[str(ei)]
        df = pd.read_parquet(src / "data" / "chunk-000" / f"episode_{ei:06d}.parquet")
        n = len(df)
        keep = np.ones(n, dtype=bool)

        trim_lo = det["boundaries"]["leading_trim"] if apply_trim["bounds"] else 0
        trim_hi = det["boundaries"]["trailing_trim"] if apply_trim["bounds"] else 0
        if trim_lo:
            keep[:trim_lo] = False
        if trim_hi:
            keep[n - trim_hi:] = False

        hes_cut = []
        if apply_trim["hesitation"]:
            for g in det["idle_gaps"]:
                if g["kind"] == "hesitation":
                    keep[g["start"]:g["end"]] = False
                    hes_cut.append([g["start"], g["end"]])

        # blur: mark only — never auto-delete
        n1 = int(keep.sum())
        df2 = df.iloc[keep].reset_index(drop=True)
        df2["frame_index"] = np.arange(n1, dtype=np.int64)
        df2["timestamp"] = (np.arange(n1, dtype=np.float64) / cfg["fps"]).astype(np.float32)
        df2["index"] = np.arange(gidx, gidx + n1, dtype=np.int64)
        gidx += n1
        total += n1
        df2.to_parquet(dst / "data" / "chunk-000" / f"episode_{ei:06d}.parquet", index=False)

        new_eps.append({"episode_index": ei, "tasks": e["tasks"], "length": n1})
        st = recompute_stats(df2, state_keys)
        for vk in video_keys:
            st[vk] = {"note": "recompute after video repair"}
        new_stats.append({"episode_index": ei, "stats": st})

        # map markers into final coords
        cum = np.cumsum(keep) - 1

        def mp(f):
            if f is None:
                return None
            f = int(max(0, min(int(f), n - 1)))
            return int(cum[f]) if keep[f] else None

        phase_rows.append({
            "episode_index": ei,
            "length_final": n1,
            "frames_final": {
                "base_start": 0 if trim_lo == det["boundaries"]["leading_trim"] else mp(0),
                "hesitation": mp(det["hesitation_frame"]),
                "arm_end": mp(det["arm_end_frame"]),
            },
            "frames_original": {
                "hesitation": det["hesitation_frame"],
                "arm_end": det["arm_end_frame"],
            },
            "marks": {
                "idle_gaps": det["idle_gaps"],
                "blur_segments": det["blur_segments"],
            },
            "trimmed": {
                "leading": int(trim_lo), "trailing": int(trim_hi),
                "hesitation_gaps": hes_cut,
            },
        })
        video_manifest["episodes"][str(ei)] = dict(
            drop_prefix=int(trim_lo), drop_suffix=int(trim_hi),
            drop_intervals=hes_cut, old_length=n, new_length=n1,
            status="pending_video_repair",
        )
        print(f"  ep{ei:03d}: {n} → {n1}  "
              f"lead_trim={trim_lo} trail_trim={trim_hi} "
              f"hes_gaps_trimmed={len(hes_cut)} "
              f"blur_marked={len(det['blur_segments'])}")

    info2 = dict(info)
    info2["total_frames"] = total
    info2["total_episodes"] = len(new_eps)
    (dst / "meta" / "info.json").write_text(json.dumps(info2, indent=2) + "\n")
    (dst / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in new_eps) + "\n")
    (dst / "meta" / "tasks.jsonl").write_text((src / "meta" / "tasks.jsonl").read_text())
    (dst / "meta" / "episodes_stats.jsonl").write_text("\n".join(json.dumps(r) for r in new_stats) + "\n")
    (dst / "meta" / "phases.jsonl").write_text("\n".join(json.dumps(r) for r in phase_rows) + "\n")
    (dst / "video_manifest.json").write_text(json.dumps(video_manifest, indent=1) + "\n")
    # carry over any extra meta files the source may have (outcomes, returns, ...)
    for p in sorted((src / "meta").iterdir()):
        if p.is_file() and p.name not in {
            "info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl",
            "cleaning_report.json",
        }:
            try:
                import shutil as _sh
                _sh.copy2(p, dst / "meta" / p.name)
            except Exception:
                pass
    return total


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="LeRobot v2.1 dataset cleaner")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=None, help="output dataset root (required with --apply)")
    ap.add_argument("--report", action="store_true", help="detect & write report only")
    ap.add_argument("--apply", action="store_true", help="write cleaned dataset to --dst")
    ap.add_argument("--trim-hesitation", action="store_true",
                    help="also delete segments marked as hesitation (default: mark only)")
    ap.add_argument("--no-trim-bounds", action="store_true",
                    help="do not trim leading/trailing static (mark only)")
    ap.add_argument("--blur-threshold", type=float, default=300.0,
                    help="Laplacian-variance threshold for blur marking. "
                         "Calibrated at 160x90 decode (this dataset scores 600-2000 = sharp; "
                         "e.g. <300 marks severe blur). Default 300.")
    ap.add_argument("--min-idle-sec", type=float, default=1.0)
    ap.add_argument("--min-blur-sec", type=float, default=0.5)
    ap.add_argument("--joint-thr", type=float, default=2.5e-3,
                    help="motion threshold in normalized units (fraction of each "
                         "dim's range per frame); robot-agnostic. Default 2.5e-3")
    ap.add_argument("--agv-thr", type=float, default=1e-4)
    ap.add_argument("--bg-thr", type=float, default=1.2)
    ap.add_argument("--max-boundary-trim-sec", type=float, default=5.0,
                    help="cap on leading/trailing trim when base motion is not observable")
    ap.add_argument("--max-video-frames", type=int, default=2000)
    ap.add_argument("--skip-video", action="store_true", help="do not decode videos")
    args = ap.parse_args()

    if not args.report and not args.apply:
        args.report = True
    if args.apply and not args.dst:
        ap.error("--apply requires --dst")
    if args.dst and Path(args.dst).exists() and args.apply:
        ap.error(f"--dst exists: {args.dst}")

    src = Path(args.src)
    info = json.loads((src / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    names = info["features"]["observation.joint_positions"]["names"]
    fps = float(info.get("fps", FPS_DEFAULT))

    cfg = dict(fps=fps, joint_names=names,
               joint_thr=args.joint_thr, agv_thr=args.agv_thr, bg_thr=args.bg_thr,
               blur_thr=args.blur_threshold, min_idle_sec=args.min_idle_sec,
               min_blur_sec=args.min_blur_sec,
               max_boundary_trim=int(args.max_boundary_trim_sec * fps))

    print(f"clean_lerobot_dataset  src={src}")
    print(f"  policy: bounds_trim={'off' if args.no_trim_bounds else 'on'}  "
          f"hesitation={'trim' if args.trim_hesitation else 'mark-only'}  "
          f"blur=mark-only  fps={fps}")

    detections = {}
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    cam_key = video_keys[0] if video_keys else None

    for e in episodes:
        ei = e["episode_index"]
        df = pd.read_parquet(src / "data" / "chunk-000" / f"episode_{ei:06d}.parquet")
        vid = None
        vnote = "skipped"
        if cam_key and not args.skip_video:
            mp4 = src / "videos" / "chunk-000" / cam_key / f"episode_{ei:06d}.mp4"
            if mp4.exists():
                vid, vnote = decode_video_gray(mp4, max_frames=args.max_video_frames)
            else:
                vnote = "missing"
        det = detect_episode(df, vid, cfg)
        det["video_note"] = vnote
        detections[str(ei)] = det
        hs = det["hesitation_frame"]
        print(f"  ep{ei:03d}: len={det['length']} video={vnote} "
              f"lead={det['boundaries']['leading_static']} trail={det['boundaries']['trailing_static']} "
              f"hes@{hs} gaps={len(det['idle_gaps'])} "
              f"(travel={sum(1 for g in det['idle_gaps'] if g['kind']=='travel_hold')}) "
              f"blur_seg={len(det['blur_segments'])}")

    # ---- report
    lead = np.array([d["boundaries"]["leading_static"] for d in detections.values()], float)
    trail = np.array([d["boundaries"]["trailing_static"] for d in detections.values()], float)
    hes = [d["hesitation_frame"] for d in detections.values() if d["hesitation_frame"] is not None]
    n_hes_gaps = sum(1 for d in detections.values() for g in d["idle_gaps"] if g["kind"] == "hesitation")
    n_travel = sum(1 for d in detections.values() for g in d["idle_gaps"] if g["kind"] == "travel_hold")
    n_blur = sum(len(d["blur_segments"]) for d in detections.values())

    report = {
        "src": str(src),
        "policy": {
            "bounds": "trim" if not args.no_trim_bounds else "mark",
            "hesitation": "trim" if args.trim_hesitation else "mark",
            "blur": "mark",
        },
        "summary": {
            "episodes": len(detections),
            "leading_static_frames_mean": float(lead.mean()),
            "leading_static_frames_max": float(lead.max()),
            "trailing_static_frames_mean": float(trail.mean()),
            "trailing_static_frames_max": float(trail.max()),
            "hesitation_frame_sec_mean": float(np.mean(hes) / fps) if hes else None,
            "hesitation_frame_sec_std": float(np.std(hes) / fps) if hes else None,
            "hesitation_gaps_total": n_hes_gaps,
            "travel_hold_gaps_kept": n_travel,
            "blur_segments_marked": n_blur,
            "base_motion_observable": sum(1 for d in detections.values() if d["can_see_base_motion"]),
            "agv_channel_used": sum(1 for d in detections.values() if d["agv_channel_used"]),
        },
        "episodes": detections,
    }

    out_report = (Path(args.dst) if args.dst else src) / "meta" / "cleaning_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=1))
    print()
    print("=== SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print(f"  report → {out_report}")

    if args.apply:
        total = write_dataset(src, Path(args.dst), info, episodes, detections, cfg,
                              apply_trim=dict(bounds=not args.no_trim_bounds,
                                              hesitation=args.trim_hesitation))
        print(f"\ncleaned dataset → {args.dst}   total_frames={total}")
        print("  meta/phases.jsonl        启/停/迟疑/模糊标记（最终+原始坐标）")
        print("  video_manifest.json      视频修复后按此裁剪对齐")
        print("  NOTE: 图像统计需在视频修复后重算；本脚本不改写视频文件。")


if __name__ == "__main__":
    main()
