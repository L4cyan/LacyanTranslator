# Lacyan Translator

**Live on-screen translation for games and apps, running entirely on your own PC.**

Lacyan Translator watches your screen, finds Chinese text, translates it with a local AI model, and paints the
translation right where the original was, over a soft blur that hides the original text. Nothing is
injected into the game, nothing is sent to the cloud, and there's nothing to configure: start it and play.

![Before and after](docs/demo-dialogue.png)

## Why another screen translator?

Most OCR translators make you draw a box, show results in a separate window, and re-translate the
same menu labels over and over. Lacyan Translator is built around three ideas:

- **It finds the text for you.** The whole window is scanned on the GPU. Lines that belong together
  are merged into blocks, a block grows as more text appears, and text that's still "typing out"
  isn't translated until it settles.
- **The translation replaces the original.** Each block gets a feathered blur of the scene behind it,
  the original text colour is kept when it's readable, and boxes grow to fit longer English without
  covering other text on screen.
- **It moves with the screen.** Translations follow scrolling text frame by frame, vanish the instant
  their text is covered by a popup or replaced, and come back the moment it reappears.
- **It only covers the original text.** Each translation stays inside the area of the Chinese it
  replaces; the font shrinks to fit. Vertical text gets one word per line, top to bottom.
- **It waits for dialogue to finish.** Text that's still typing out is only translated once it has
  stopped changing, so you never get half a sentence.
- **It's always readable.** Every translation gets a solid outline and soft shadow, keeping the game's
  own text colour when that colour stands out.
- **It survives busy pages.** Built and tested against a dense shopping page (60+ text blocks per
  screen): short phrases are translated in streamed batches top to bottom, and a new page clears the
  old translations on the very first frame.
- **It remembers.** Every translated phrase is stored in a local translation memory and shown instantly
  the next time the exact same phrase appears. Shopping and game UI phrasebooks cover common words.

## Install

1. Download **[LacyanTranslator-Setup.exe](https://github.com/L4cyan/LacyanTranslator/releases/latest/download/LacyanTranslator-Setup.exe)**
   from the [latest release](https://github.com/L4cyan/LacyanTranslator/releases/latest).
2. Run it and click **Install**. That's it.

The setup installs everything for you: its own private copy of Python and every component, GPU
acceleration for your graphics card, [Ollama](https://ollama.com) if you don't have it, and the
translation model. No administrator rights are needed, and it doesn't touch any Python you already have.

You choose how much to download:

| Option | Download | |
| --- | --- | --- |
| **Fastest model only** (default) | about 1.1 GB | The fast model (Q4); quick and good for games and shops |
| Install all model sizes | about 4.5 GB | Fast, Balanced and Best quality, switchable any time in the app |

NVIDIA users can also tick CUDA acceleration (about 1.5 GB more) for the fastest text reading; everyone
else gets DirectML, which works on any GPU. Uninstall any time from Windows Settings → Apps.

Windows SmartScreen may warn about an unknown publisher, because the setup isn't code-signed yet:
click **More info → Run anyway**.

On first launch it checks your hardware (GPU, graphics memory, CPU, RAM) and recommends a performance
profile, which you confirm:

| Profile | Follows text at | Translation | Best for |
| --- | --- | --- | --- |
| Low | 20 fps | 1 stream, small batches | Laptops without a dedicated GPU |
| Balanced | 30 fps | 1 stream | Most gaming PCs |
| High | 60 fps | 2 parallel streams | NVIDIA GPUs with 8 GB+ and 16 GB RAM |

Each launch then opens a small window: pick the language to translate **to** (English by default), the
active window or the whole screen, and the performance profile, then press **Start translating**.

Run your game in **windowed** or **borderless** mode. Exclusive fullscreen can hide overlays.

## Using it

Lacyan Translator lives in the system tray (the green **译** icon). Click it to change languages or
options while it runs.

| Hotkey | Action |
| --- | --- |
| `Alt+T` | Show the original text / show translations |
| `Alt+P` | Pause / resume |
| `Ctrl+Alt+Q` | Quit |

The tray menu lets you change the target language, switch between translating the **active window**
or the **whole screen**, open the settings file, and clear the translation memory.

Target languages: English, Filipino, Indonesian, Japanese, Korean, Vietnamese, Thai, Malay, Spanish,
Portuguese, French, German, Italian, Russian, Arabic, Turkish, Hindi, Traditional Chinese.

## Glossaries and phrasebooks

- **Glossaries** (`glossaries/xianxia.txt`): terms the model must translate a specific way, such as
  `筑基 = Foundation Establishment`. Only terms that actually appear in a line are sent with it.
- **Phrasebooks** (`glossaries/ui.txt`): exact matches that skip the model entirely. Short menu words
  are where machine translation guesses wrong (`确定` is "OK", not "Confirmed").

One `source = translation` pair per line. Add your own files in `config.json`.

## Settings

`config.json` is created next to `LacyanTranslator.bat` on first run. The most useful settings:

| Setting | Default | What it does |
| --- | --- | --- |
| `target_language` | `English` | Language to translate into |
| `endpoint`, `model`, `api_key` | local Ollama, `lacyan-mt` | Any OpenAI-compatible server works: LM Studio, llama.cpp, or a cloud API |
| `capture` | `foreground` | `foreground` = the active window, `monitor` = the whole screen |
| `font_family` | `Segoe UI` | Font for translations |
| `blur_strength`, `backdrop_tint` | `1.0`, `0.35` | How strongly the original is hidden |
| `keep_original_color` | `true` | Reuse the game's text colour when it's readable |
| `max_grow` | `1.0` | `1.0` keeps every translation inside the original text's area (the font shrinks to fit); higher lets boxes grow into free space |
| `hide_from_capture` | `true` | Keep the overlay out of screenshots and recordings (this is also how Lacyan Translator avoids reading its own output) |

## How it works

```
screen capture ─▶ change detection ─▶ GPU OCR ─▶ block grouping ─▶ tracking ─▶ translation memory ─▶ local model
                                                                                        │
                        click-through overlay ◀── blur + fitted text ◀──────────────────┘
```

- **Capture**: DXGI Desktop Duplication (`dxcam`, about 0.1 ms per frame, and only when the screen
  actually changed), up to the profile's frame rate, of the active window. A still screen costs almost
  nothing. The overlay window is excluded from capture (`WDA_EXCLUDEFROMCAPTURE`), so Lacyan Translator
  always sees the real game underneath.
- **Following**: every frame, each shown translation checks that its text is still there with a small
  normalized-correlation match (about 3 ms for the whole frame). Moved text is found again nearby, with
  the scroll speed used to predict where it went; text that's gone hides on the same frame.
- **OCR**: PP-OCRv4 through RapidOCR on ONNX Runtime (CUDA on NVIDIA, DirectML on other GPUs, CPU as
  a last resort), on its own thread so following never waits for it. Text boxes are detected on the full frame, but only boxes in the area that changed
  are read again; everything else reuses the previous reading.
- **Grouping and tracking**: OCR lines are merged into paragraphs, matched across frames so the
  overlay doesn't flicker, and translated only once their text has been stable for two passes.
- **Translation**: Hunyuan-MT 1.5 (1.8B) through Ollama. Short phrases go in numbered, streamed
  batches (each translation appears as soon as its line arrives); long ones go alone. Exact phrasebook
  and memory hits never reach the model. Lines that come back untranslated are retried.
- **Rendering**: a click-through, always-on-top Qt window that repaints only the areas that changed.
  All backdrops are drawn first and then all text, so one block's blur can never cover another
  block's translation.

Measured on an RTX 4070 laptop with a dense shopping page at 2560×1600: translations stay on their text
while scrolling at 2,400 px/s (0 px error), a cold screen of 44 blocks fills in about 7 s (first lines
after 2 s), a new page clears the old translations on the first frame, and per-frame work is about 9 ms
while scrolling and 0.3 ms on a still screen.

## Limits

- Source language is Chinese for now. The OCR model also reads English, so mixed screens work.
- Screen OCR can't see text that isn't drawn yet, and very stylised fonts may be misread.
- Windows 10 2004 or newer is needed to hide the overlay from capture.

## Developing

Run from source with `LacyanTranslator.bat` (it sets up a local environment on first run), or:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt "onnxruntime-gpu[cuda,cudnn]"
.venv\Scripts\python -m lacyan_translator                                # run the tray app
.venv\Scripts\python -m lacyan_translator --snapshot shot.png out.png    # translate a screenshot to a file
```

Logs go to `data/lacyan.log`; the translation memory is `data/translations.sqlite`.
Build the installer with `python installer/build.py` (output: `installer/dist/LacyanTranslator-Setup.exe`).

## License

MIT
