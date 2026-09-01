#!/usr/bin/env python3
"""Generate Nano Banana (gemini-3-pro-image-preview) polished versions of the
PC-TikiTorch schematics. Input: sketch PNG + restyle prompt. Output: <name>_nb.png."""
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

GFX = os.path.dirname(os.path.abspath(__file__))

# --- load .env ---
env = {}
with open(os.path.join(GFX, ".env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
API_KEY = env["NANOBANANA_API_KEY"]
MODEL = env.get("NANOBANANA_MODEL", "gemini-3-pro-image-preview")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PREAMBLE = (
    "Redraw this technical schematic as a polished graphic. Preserve every text label, "
    "number, and arrow direction exactly as in the input image - do not invent, drop, or "
    "paraphrase any text. Keep the same layout and information hierarchy. Dark navy "
    "(#0B1120) background, consistent family style across the series: crisp vector-poster "
    "look, subtle glow on accent colors (cyan = forward/data, amber = error/backward, "
    "violet = predictive coding, green = measured results), thin light-gray connector "
    "lines, generous margins, no watermark. Landscape, high resolution. "
)

FIGS = {
    "01_big_picture": (
        "Style it as the opening slide of a hardware-startup keynote. Give the three pillar "
        "cards subtle depth and an icon each: a max/plus lattice for tropical arithmetic, a "
        "neuron with a small error-loop for predictive coding, and a silicon die for the FPGA "
        "fabric. Make the green 'ON SILICON TODAY - MEASURED' band feel like a certification "
        "stamp, and render the north-star box as a faint constellation/summit motif. The title "
        "'PC-TikiTorch' may get a small tiki-torch flame mark whose flame is a tropical "
        "max-plus curve."
    ),
    "02_backprop_vs_pc": (
        "Emphasize the contrast: the backprop side slightly desaturated and heavy, with the "
        "long red gradient chain rendered as a thick tangled cable wrapping around the whole "
        "stack; the predictive-coding side clean and airy, with short elegant one-hop arrows "
        "and small glowing epsilon error units. Keep the cross/check annotation lists crisp. "
        "The bottom banner is the punchline - give it weight."
    ),
    "03_fabric": (
        "Render the eight FPGAs as photorealistic-ish chip packages on a dark PCB, connected "
        "left to right, with the cyan hidden-state arrows and amber error arrows as glowing "
        "traces between them. The host is a plainer server box on the left. Keep the green "
        "MEASURED band at the bottom verbatim. CRITICAL - the 'Bubble-free 1F1B pipeline' "
        "panel is DATA and must be copied cell-for-cell as a strict flat grid of rounded "
        "rectangles with NO arrows between cells and NO stylization. It has exactly 10 time "
        "columns and 4 rows; row 'chip 0' starts at column 1, 'chip 1' at column 2, 'chip 2' "
        "at column 3, 'chip 3' at column 4 (a staircase). Exact cell contents: "
        "chip 0: F1 F2 F3 F4 B1 F5 B2 F6 B3 F7; "
        "chip 1: F1 F2 F3 B1 F4 B2 F5 B3 F6; "
        "chip 2: F1 F2 B1 F3 B2 F4 B3 F5; "
        "chip 3: F1 B1 F2 B2 F3 B3 F4. "
        "F cells cyan, B cells amber. There is no B4 and no duplicated cell anywhere. "
        "Row offsets matter: each row starts exactly ONE column later than the row above. "
        "SPELLING: every chip is labeled 'mult-free fwd' (m-u-l-t), never 'muit-free'."
    ),
    "04_inside_chip": (
        "Style as an annotated die shot / datasheet cutaway: the chip outline becomes a "
        "silicon die with faint standard-cell texture, the three lanes (cyan forward, amber "
        "backward, violet optimizer) as distinct metal-layer routing regions. Keep the "
        "resource-utilization bars as exact horizontal bars with their percentages. The green "
        "invariant banner should read like a stamped datasheet warning: 'DSP_MULT = 0'."
    ),
    "05_scaling_ladder": (
        "Render as a mountain-route infographic: the green measured path as a solid climbing "
        "rope with carabiner nodes, the amber segment glowing (in progress), the dashed gray "
        "section as an unclimbed ridge in fog, the violet flag at the summit. Milestone cards "
        "become small route-log placards pinned to the rope. Keep the throughput bar-chart "
        "inset and the honesty ribbon at the bottom exactly as-is."
    ),
    "06_path_to_100b": (
        "THIS IS A DATA-DENSE CHART: the text and numbers ARE the payload. Reproduce every "
        "single string, number, and tag EXACTLY as in the input - transcribe, do not redraw "
        "from memory; if unsure about any character, copy it. Do NOT change any digit, unit, "
        "or word (e.g. 'D=10240', '100.7B', '2T', '$0.4B', '330k tokens/step' must appear "
        "verbatim). Restyle ONLY the container aesthetics: subtle glow on card borders (green "
        "for the memory card, amber for the risk card and wall panel, violet for the target), "
        "a faint circuit texture in the background, slightly richer typography. Keep the "
        "exact same layout, same line breaks, same table columns."
    ),
}


def generate(name, prompt, out_path, attempts=4):
    SIZE = "4K" if name == "06_path_to_100b" else "2K"
    with open(os.path.join(GFX, f"{name}.png"), "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    body = {
        "contents": [{
            "parts": [
                {"text": PREAMBLE + prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            ]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "16:9", "imageSize": SIZE},
        },
    }
    data = json.dumps(body).encode()
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            URL,
            data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:600]
            print(f"[{name}] attempt {attempt}: HTTP {e.code}: {err}", flush=True)
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(20 * attempt)
                continue
            return False
        except Exception as e:  # timeouts etc.
            print(f"[{name}] attempt {attempt}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(15)
                continue
            return False
        parts = (payload.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        for part in parts:
            blob = part.get("inline_data") or part.get("inlineData")
            if blob and blob.get("data"):
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(blob["data"]))
                print(f"[{name}] OK -> {out_path}", flush=True)
                return True
        print(f"[{name}] attempt {attempt}: no image part; keys={list(payload)}; "
              f"finish={(payload.get('candidates') or [{}])[0].get('finishReason')}", flush=True)
        if attempt < attempts:
            time.sleep(10)
    return False


if __name__ == "__main__":
    targets = sys.argv[1:] or [n for n in FIGS if os.path.exists(os.path.join(GFX, f"{n}.png"))]
    failed = []
    for n in targets:
        out = os.path.join(GFX, f"{n}_nb.png")
        if not generate(n, FIGS[n], out):
            failed.append(n)
    if failed:
        print(f"FAILED: {failed}", flush=True)
        sys.exit(1)
    print("ALL DONE", flush=True)
