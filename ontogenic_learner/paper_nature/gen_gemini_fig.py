#!/usr/bin/env python3
"""Generate a conceptual figure with Gemini (Nano Banana Pro) and save PNG.

Usage: python gen_gemini_fig.py <prompt_file_or_-> <out.png> [aspect]
Reads GEMINI_API_KEY / NANOBANANA_MODEL from ./.env . No third-party deps.
"""
import base64
import json
import pathlib
import sys
import urllib.request
import urllib.error

HERE = pathlib.Path(__file__).parent


def env(key, default=None):
    for line in (HERE / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return default


def generate(prompt, out_path, aspect="16:9", model=None):
    key = env("GEMINI_API_KEY") or env("NANOBANANA_API_KEY")
    model = model or env("NANOBANANA_MODEL", "gemini-3-pro-image-preview")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": aspect, "imageSize": "2K"},
        },
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:2000])
        return False
    # dig out inline image data
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        print("Unexpected response:", json.dumps(data)[:2000])
        return False
    for p in parts:
        blob = p.get("inlineData") or p.get("inline_data")
        if blob and blob.get("data"):
            pathlib.Path(out_path).write_bytes(base64.b64decode(blob["data"]))
            print("wrote", out_path, "mime", blob.get("mimeType") or blob.get("mime_type"))
            return True
        if p.get("text"):
            print("model text:", p["text"][:500])
    print("no image in response")
    return False


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    aspect = sys.argv[3] if len(sys.argv) > 3 else "16:9"
    prompt = sys.stdin.read() if src == "-" else pathlib.Path(src).read_text()
    ok = generate(prompt, out, aspect)
    sys.exit(0 if ok else 1)
