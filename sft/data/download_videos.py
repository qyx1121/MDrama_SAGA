"""Download the M-Drama video clips referenced by the released annotations.

The HF dataset yixin1121/M-Drama ships annotations only; each clip is defined
by (url, start_time, end_time) pointing into a YouTube video. This script uses
yt-dlp (--download-sections, requires ffmpeg) to fetch each clip and save it
as {output_dir}/{video_id}.mp4, matching the paths written by make_sft_data.py.

Usage:
    python data/download_videos.py --output_dir data/videos [--workers 8]

Note: source videos belong to their respective copyright holders (see the
dataset license, CC BY-NC 4.0). Download for research use only.
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset


def download_clip(video_id, url, start_time, end_time, output_dir):
    out_path = os.path.join(output_dir, video_id + ".mp4")
    if os.path.exists(out_path):
        return video_id, "skipped"
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        "-f", "best[ext=mp4]/best",
        "--no-playlist",
        "--quiet", "--no-warnings",
        "-o", out_path,
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        return video_id, "ok"
    except subprocess.CalledProcessError as e:
        return video_id, f"failed: {e.stderr.decode(errors='ignore')[-200:]}"
    except subprocess.TimeoutExpired:
        return video_id, "failed: timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yixin1121/M-Drama")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="data/videos")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ds = load_dataset(args.dataset, split=args.split)

    # one clip per video_id
    clips = {}
    for row in ds:
        if row["video_id"] not in clips:
            clips[row["video_id"]] = (row["url"], row["start_time"], row["end_time"])
    print(f"{len(clips)} unique clips to download")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_clip, vid, url, st, et, args.output_dir): vid
            for vid, (url, st, et) in clips.items()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            vid, status = fut.result()
            if status.startswith("failed"):
                failed.append((vid, status))
            if i % 100 == 0:
                print(f"[{i}/{len(clips)}] done, {len(failed)} failed")

    print(f"Finished: {len(clips) - len(failed)} ok/skipped, {len(failed)} failed")
    if failed:
        with open(os.path.join(args.output_dir, "failed.txt"), "w") as f:
            for vid, status in failed:
                f.write(f"{vid}\t{status}\n")
        print("Failed ids written to failed.txt — re-run to retry")


if __name__ == "__main__":
    main()
