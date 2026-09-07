# RL data preparation

Builds `data/rl/mdrama_rl_graph/train.parquet` — the RL training set consumed by
`train/qwen3_vl_8b_drama_dapo_graph.sh` — from the released
[M-Drama](https://huggingface.co/datasets/yixin1121/M-Drama) annotations.

## What the source gives you

| | |
|---|---|
| 32,361 QA annotations | over 8,102 short-drama clips (~52 GB) |
| `stage == "SFT+RL"` | the 14,135 annotations used for RL (the rest is SFT-only) |
| videos | **not** distributed — each clip is `(url, start_time, end_time)` into a YouTube video |

You therefore need three things: the annotations (step 1), the clips (step 2),
and the ground-truth scene graphs (step 3).

## Output schema

Identical to EasyR1's RL parquet, one row per annotation:

| column | type | content |
|---|---|---|
| `videos` | `list[str]` | absolute path of the clip |
| `system` | `str` | system prompt, read from `train/format_prompt/drama.jinja` |
| `problem` | `str` | `<video>\n{question}` (+ `\nOptions: [A. ...; B. ...]` for MC) |
| `answer` | `str` | `<caption>…</caption>\n<think>…</think>\n<answer>…</answer>` |
| `task_type` | `str` | `MC` / `OE` / `SUMMARY` |
| `gt_graph` | `str` | JSON scene graph of the ground-truth caption |

`task_type` is derived from the annotation: answer ∈ {A,B,C,D} → `MC`; the canned
caption question → `SUMMARY`; otherwise `OE`.

## Quick start

```bash
pip install -r ../../requirements.txt          # datasets, pandas, pyarrow
huggingface-cli login                          # the M-Drama repo is gated

# fetch the paper's ground-truth scene graphs (optional but recommended)
GT_GRAPHS=$(python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('yixin1121/M-Drama', 'gt_graphs.json', repo_type='dataset'))")

bash run.sh --video_dir /path/to/clips --cap2graph "${GT_GRAPHS}"
```

## Step by step

### 1. Annotations

```bash
python prepare_rl_data.py download             # -> mdrama_annotations.jsonl
```

Pulls `yixin1121/M-Drama` from the Hub and keeps only `stage == "SFT+RL"`
(14,135 rows). Use `--stage all` for the full 32,361.

### 2. Clips

M-Drama ships no videos. Two supported setups:

**(a) Download them** — one clip per `video_id`, named `{video_id}.mp4`:

```bash
python ../../sft/data/download_videos.py --output_dir ../../data/videos
python prepare_rl_data.py build --video_dir ../../data/videos
```

This is the recommended route: the SFT stage (`sft/data/make_sft_data.py`) uses
the exact same paths, so you only download once.

**(b) Reuse an existing clip library** whose files are named
`{title}_{HH:MM:SS}-{HH:MM:SS}.mp4` — resolve the names once and cache the map:

```bash
python prepare_rl_data.py video-map --video_dir /path/to/clips --mode timestamp
python prepare_rl_data.py build --video_dir /path/to/clips \
                                --cap2graph /path/to/caption2graph.json
```

`video-map` indexes the directory by the timestamps embedded in the filenames.
A handful of clips share the same `(start, end)` offsets across different source
videos and cannot be disambiguated that way; they are listed in
`video_map.json.unresolved.json` for manual fixing. If you have an SFT json
(`sft/data/make_sft_data.py` output) that already points at those clips, use
`--mode caption --sft_json <file>` instead — the caption is a unique join key and
resolves everything.

### 3. Ground-truth scene graphs

`gt_graph` is the target of SAGA's graph-alignment reward: a
Character / Prop / Scene / Event graph parsed from the ground-truth caption by an
LLM, using the ontology in `train/system_prompts/graph_build_prompt.txt`. The
reward's node-type thresholds (`train/reward_function/reward_graph/config.py`) are
tuned for that ontology, so the cache must be built with that prompt.

The cache is a plain json dict, either keyed by the caption text or by
`video_id`:

```json
{
  "<caption text>": {"nodes": [{"id": "n1", "type": "Scene", "name": "Office"}],
                     "edges": [{"source": "n2", "relation": "AGENT_OF", "target": "n4"}]}
}
```

Pass it with `--cap2graph`. Without it the parquet is still produced, but
`gt_graph` is empty and `global_caption_rewards` stays at 0 — i.e. you lose the
graph reward and keep only accuracy + format.

**Pre-built graphs (recommended).** The exact graphs used for the paper
(Qwen3-14B, `graph_build_prompt.txt`) ship with the dataset as `gt_graphs.json`,
keyed by `video_id` — `prepare_rl_data.py build` accepts it as-is:

```bash
GT_GRAPHS=$(python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('yixin1121/M-Drama', 'gt_graphs.json', repo_type='dataset'))")

python prepare_rl_data.py build --video_dir /path/to/clips --cap2graph "${GT_GRAPHS}"
```

To rebuild the cache yourself instead, send the 8,102 unique captions (one call
per caption, `mdrama_annotations.jsonl` → `caption` column) to a vLLM
OpenAI-compatible endpoint with `train/system_prompts/graph_build_prompt.txt`
as the instruction, and dump the parsed graphs as `{caption.strip(): graph}`.

> One graph per *clip*, not per annotation — the 14,135 RL annotations share only
> 8,102 distinct captions, so keying by caption keeps the cache (and the LLM bill)
> at 8,102 entries.

## Notes

* Rows whose clip is missing on disk are dropped with a warning; add `--strict`
  to fail instead.
* `system` is read straight out of `train/format_prompt/drama.jinja`, so the data
  cannot drift from the chat template used at rollout time.
* The annotations are bilingual (roughly a third of the captions are Chinese);
  they are passed through verbatim. Only the *graphs* are forced to English, by
  the graph-construction prompt.
* Source videos belong to their copyright holders — download for research use
  only (dataset license: CC BY-NC 4.0).
