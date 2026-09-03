import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import requests
from openai import OpenAI

RSS_URL = "https://feeds.acast.com/public/shows/631a89913c2be9001415dc41"
STATE_PATH = Path("state/latest.json")
REPORT_DIR = Path("reports")
AUDIO_DIR = Path(".tmp")


def get_latest_episode():
    r = requests.get(RSS_URL, timeout=30, headers={"User-Agent": "morningcall-transcribera/1.0"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    channel = root.find("channel")
    if channel is None:
        raise RuntimeError("RSS channel not found")
    item = channel.find("item")
    if item is None:
        raise RuntimeError("No episode found in RSS")

    def text(tag):
        el = item.find(tag)
        return (el.text or "").strip() if el is not None else ""

    enclosure = item.find("enclosure")
    if enclosure is None or not enclosure.attrib.get("url"):
        raise RuntimeError("Latest episode has no audio enclosure URL")

    return {
        "guid": text("guid") or enclosure.attrib["url"],
        "title": text("title"),
        "pub_date": text("pubDate"),
        "description": text("description"),
        "audio_url": enclosure.attrib["url"],
        "length": enclosure.attrib.get("length", ""),
    }


def load_state():
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(episode):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"guid": episode["guid"], "title": episode["title"], "pub_date": episode["pub_date"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_filename(value):
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return value[:100] or "morning-call"


def download_audio(url):
    AUDIO_DIR.mkdir(exist_ok=True)
    path = AUDIO_DIR / "latest.mp3"
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "morningcall-transcribera/1.0"}) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path


def transcribe(audio_path):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            language="en",
        )
    return result.text


def create_japanese_report(episode, transcript):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""
You are preparing a Japanese investor briefing from the latest NAB Morning Call podcast.
Do NOT reproduce the full verbatim transcript or provide a line-by-line translation. Instead,
produce a detailed Japanese rendering that preserves the episode's logical flow and all material
market/investment information, while summarizing the spoken content in your own words.

Episode title: {episode['title']}
Publication date: {episode['pub_date']}

Requirements:
1. Start with a short executive summary (3-6 bullets).
2. Then follow the episode in roughly the same order as spoken, using clear section headings.
3. Cover all material points on oil, inflation, bonds, central banks, FX, Australia/RBA, US rates,
   Europe and other markets mentioned.
4. For each market point, explain why it matters for asset prices and the likely transmission mechanism.
5. Preserve important numbers, dates, names, forecasts and directional views.
6. Financial terminology should be natural Japanese; include the English term in parentheses when useful.
7. Clearly distinguish NAB's view from factual observations.
8. End with "投資家向けチェックポイント" containing 5-10 actionable items to monitor.
9. Do not invent facts that are absent from the audio.

Audio transcript (source material):
{transcript}
"""
    response = client.responses.create(model="gpt-5.6-luna", input=prompt)
    return response.output_text


def main():
    episode = get_latest_episode()
    state = load_state()
    if state.get("guid") == episode["guid"]:
        print(f"No new episode: {episode['title']}")
        return

    audio_path = download_audio(episode["audio_url"])
    transcript = transcribe(audio_path)
    report = create_japanese_report(episode, transcript)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_part = episode["pub_date"][:16].replace(":", "-") if episode["pub_date"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = safe_filename(f"{date_part}-{episode['title']}") + ".md"
    output = f"""# NAB Morning Call — {episode['title']}\n\n- **公開日:** {episode['pub_date']}\n- **音声:** {episode['audio_url']}\n\n> This report is a detailed Japanese summary based on the episode audio. It is not a verbatim transcript or line-by-line translation.\n\n{report}\n"""
    (REPORT_DIR / filename).write_text(output, encoding="utf-8")
    save_state(episode)
    print(f"Created report: {REPORT_DIR / filename}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
