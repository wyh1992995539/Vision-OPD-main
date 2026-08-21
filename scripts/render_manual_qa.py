#!/usr/bin/env python3
"""Render the manual-QA worklist as a local, interactive HTML review page."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path


REVIEW_FIELDS = (
    "full_image_has_box",
    "crop_matches_box",
    "question_matches_region",
    "answer_matches_image",
)

FIELD_LABELS = {
    "full_image_has_box": "全图红框是否清楚且位置合理",
    "crop_matches_box": "Teacher crop 是否对应红框区域",
    "question_matches_region": "问题是否与目标区域匹配",
    "answer_matches_image": "标准答案是否与图像内容一致",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render an interactive manual-QA HTML page.")
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "artifacts" / "data" / "manual_qa_30.jsonl",
    )
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "artifacts" / "data" / "manual_qa_30.html",
    )
    return parser.parse_args()


def load_worklist(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [
                field
                for field in ("qa_index", "sample_id", "full_image_path", "crop_image_path")
                if field not in record
            ]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {missing}")
            records.append(record)
    if len(records) != 30:
        raise ValueError(f"Expected 30 manual-QA records, found {len(records)}")
    return records


def radio_group(index: int, field: str, current: object) -> str:
    yes_checked = " checked" if current is True else ""
    no_checked = " checked" if current is False else ""
    escaped_field = html.escape(field, quote=True)
    label = html.escape(FIELD_LABELS[field])
    return f"""
      <fieldset class="check-row" data-field="{escaped_field}">
        <legend>{label}</legend>
        <label><input type="radio" name="{escaped_field}-{index}" value="true"{yes_checked}> 通过</label>
        <label><input type="radio" name="{escaped_field}-{index}" value="false"{no_checked}> 有问题</label>
      </fieldset>"""


def render_card(record: dict[str, object], subset_root: Path) -> str:
    index = int(record["qa_index"])
    full_path = (subset_root / Path(str(record["full_image_path"]))).resolve()
    crop_path = (subset_root / Path(str(record["crop_image_path"]))).resolve()
    if not full_path.is_file() or not crop_path.is_file():
        raise FileNotFoundError(f"Missing QA image pair for {record['sample_id']}")
    controls = "".join(radio_group(index, field, record.get(field)) for field in REVIEW_FIELDS)
    return f"""
    <article class="card" id="sample-{index}" data-index="{index - 1}">
      <header>
        <h2>#{index} · {html.escape(str(record['split']))} · {html.escape(str(record['area_bucket']))}</h2>
        <code>{html.escape(str(record['sample_id']))}</code>
        <span class="status">pending</span>
      </header>
      <div class="images">
        <figure><a href="{html.escape(full_path.as_uri(), quote=True)}" target="_blank"><img loading="lazy" src="{html.escape(full_path.as_uri(), quote=True)}"></a><figcaption>Student 全图</figcaption></figure>
        <figure><a href="{html.escape(crop_path.as_uri(), quote=True)}" target="_blank"><img loading="lazy" src="{html.escape(crop_path.as_uri(), quote=True)}"></a><figcaption>Teacher crop</figcaption></figure>
      </div>
      <section class="text">
        <h3>问题</h3><pre>{html.escape(str(record.get('problem', '')))}</pre>
        <p><strong>标准答案：</strong>{html.escape(str(record.get('answer', '')))}</p>
        <p><strong>bbox：</strong><code>{html.escape(json.dumps(record.get('bbox'), ensure_ascii=False))}</code></p>
      </section>
      <section class="review">{controls}
        <label class="note">备注<textarea rows="2" placeholder="有问题时说明具体原因">{html.escape(str(record.get('note', '')))}</textarea></label>
      </section>
    </article>"""


def build_html(records: list[dict[str, object]], subset_root: Path) -> str:
    cards = "\n".join(render_card(record, subset_root) for record in records)
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    fields = json.dumps(REVIEW_FIELDS)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vision-OPD Manual QA 30</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f3f5f7;color:#17202a}}
.toolbar{{position:sticky;top:0;z-index:5;background:#fff;padding:12px 20px;border-bottom:1px solid #ccd3da;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
button{{padding:8px 14px;border:1px solid #567;border-radius:6px;background:#fff;cursor:pointer}} button.primary{{background:#1769aa;color:#fff}}
main{{max-width:1200px;margin:20px auto;padding:0 16px}} .card{{background:#fff;border:1px solid #d6dce1;border-radius:10px;margin:18px 0;padding:16px;box-shadow:0 2px 8px #0001}}
.card header{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}} .status{{margin-left:auto;padding:4px 10px;border-radius:999px;background:#eee}}
.status.pass{{background:#d8f5df;color:#176b2c}} .status.suspected_badcase{{background:#ffe0dc;color:#9d2418}}
.images{{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-top:14px}} figure{{margin:0;text-align:center;background:#eef1f4;padding:8px;border-radius:8px}}
img{{width:100%;height:430px;object-fit:contain;background:#111}} figcaption{{padding-top:6px;font-weight:600}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:6px}} .review{{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px}}
.check-row{{border:1px solid #d6dce1;border-radius:6px}} .check-row label{{margin-right:18px}} .note{{grid-column:1/-1;display:grid;gap:5px}} textarea{{font:inherit;padding:8px}}
@media(max-width:800px){{.images,.review{{grid-template-columns:1fr}}img{{height:300px}}}}
</style>
</head>
<body>
<div class="toolbar"><strong>Vision-OPD 人工 QA</strong><span id="progress">0/30 完成</span><button id="export" class="primary">导出 manual_qa_30.jsonl</button><button id="clear">清除本页缓存</button></div>
<main>{cards}</main>
<script>
const records={payload};
const fields={fields};
const storageKey="vision-opd-manual-qa-v1";
function collect(index){{
  const card=document.querySelector(`[data-index="${{index}}"]`), record=records[index];
  fields.forEach(field=>{{const checked=card.querySelector(`input[name="${{field}}-${{index+1}}"]:checked`);record[field]=checked?checked.value==="true":null;}});
  record.note=card.querySelector("textarea").value.trim();
  const values=fields.map(field=>record[field]);
  record.status=values.every(value=>value===true)?"pass":values.some(value=>value===false)?"suspected_badcase":"pending";
  const badge=card.querySelector(".status");badge.textContent=record.status;badge.className=`status ${{record.status}}`;
}}
function save(){{records.forEach((_,i)=>collect(i));localStorage.setItem(storageKey,JSON.stringify(records));updateProgress();}}
function updateProgress(){{const done=records.filter(r=>r.status!=="pending").length;document.getElementById("progress").textContent=`${{done}}/30 完成`;}}
function restore(){{const saved=localStorage.getItem(storageKey);if(!saved)return;const values=JSON.parse(saved);values.forEach((value,index)=>{{fields.forEach(field=>{{if(value[field]===true||value[field]===false){{const input=document.querySelector(`input[name="${{field}}-${{index+1}}"][value="${{value[field]}}"]`);if(input)input.checked=true;}}}});const area=document.querySelector(`[data-index="${{index}}"] textarea`);if(area)area.value=value.note||"";}});}}
document.querySelectorAll("input,textarea").forEach(element=>element.addEventListener("change",save));
document.querySelectorAll("textarea").forEach(element=>element.addEventListener("input",save));
document.getElementById("export").addEventListener("click",()=>{{save();const text=records.map(record=>JSON.stringify(record)).join("\\n")+"\\n";const url=URL.createObjectURL(new Blob([text],{{type:"application/x-ndjson"}}));const link=document.createElement("a");link.href=url;link.download="manual_qa_30.jsonl";link.click();URL.revokeObjectURL(url);}});
document.getElementById("clear").addEventListener("click",()=>{{if(confirm("仅清除这个审查页面在浏览器中的勾选缓存？")){{localStorage.removeItem(storageKey);location.reload();}}}});
restore();records.forEach((_,i)=>collect(i));updateProgress();
</script>
</body>
</html>
"""


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    records = load_worklist(args.input.resolve())
    output = args.output.resolve()
    write_text_atomic(output, build_html(records, args.subset_root.resolve()))
    print(f"Manual QA HTML: {output}")
    print("Open it in Chrome, review all 30 samples, then export manual_qa_30.jsonl.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
