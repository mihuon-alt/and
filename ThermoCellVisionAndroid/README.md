# ThermoCell Vision — Java/Android port

Pure Java Android Studio project. No Python, no Kivy, no buildozer.

Ported 1:1 from the original `thermocell_vision` (Kivy) source:

| Original (Python/Kivy)              | This project (Java/Android)                          |
|--------------------------------------|--------------------------------------------------------|
| `main.py`                            | `MainActivity.java` (permission flow + screen switch) |
| `ui/camera_view.py` (camera4kivy)    | `MainActivity.java` (CameraX `ImageAnalysis`)          |
| `ui/thermocellvision.kv`             | `res/layout/screen_camera.xml` + `screen_permission.xml` |
| `ui/widgets.py` (GlassPanel etc.)    | `res/drawable/bg_glass_panel*.xml`, `bg_glass_button.xml`, `styles.xml` |
| `vision/heat_detector.py`            | `detection/HeatRegionDetector.java` (same 12-stage OpenCV pipeline) |
| `vision/temporal_smoother.py`        | `detection/MaskSmoother.java`, `ValueSmoother.java`   |

## AI verdict (Groq)

On top of the always-on OpenCV pipeline, `GroqVisionClassifier.java` sends
a downscaled JPEG of the current frame to Groq's vision-model endpoint
every ~3 seconds and asks for a `{hot, confidence, label}` JSON verdict.
This is shown in the new "AI VERDICT (GROQ)" panel above the bottom
status bar. It is intentionally decoupled from the per-frame OpenCV loop:
- **OpenCV (`HeatRegionDetector`)** — runs every frame, draws the red
  hot-region fill, the white surface-boundary outline, and the green
  placement-zone outline. Works fully offline, no API key needed.
- **Groq (`GroqVisionClassifier`)** — runs on a timer in the background,
  gives a higher-level "yes this is actually a hot object" sanity check
  with a confidence score and a short label. Requires network + an API
  key; if unset or a call fails, the panel just shows "NO API KEY SET" /
  "AI UNAVAILABLE" and the OpenCV overlay keeps working normally.

**Setup:** copy `local.properties.example` to `local.properties` and put
your key in `groq.apiKey=...` (get one at https://console.groq.com/keys).
This file is gitignored so the key is never committed. The model id used
is a placeholder in `GroqVisionClassifier.MODEL` — check
https://console.groq.com/docs/models for the current vision-capable model
before building, since Groq's model lineup changes over time.

Same honesty note as the original: this never measures real temperature.
A phone camera has no thermal sensor — both the OpenCV heuristic and the
Groq verdict are visual judgements, not sensor readings.

## Building

This project needs the Android SDK, NDK-free (pure Java, no NDK needed)
build tools, and Gradle — none of which are reachable from the sandbox
that generated this code. To build the APK:

**Locally (Android Studio):** open this folder in Android Studio, let it
sync Gradle, then Build → Build Bundle(s)/APK(s) → Build APK(s).

**Or via GitHub Actions (no local Android Studio needed):**
```bash
git init
git add -A
git commit -m "ThermoCell Vision - Java port"
git remote add origin https://github.com/<you>/thermocell-vision-android.git
git push -u origin main
```
The included `.github/workflows/build-apk.yml` runs `./gradlew assembleDebug`
on GitHub's runners and uploads the resulting APK as a build artifact
(Actions tab → the run → Artifacts).

## Notes / things worth checking once you can actually build & run it

- `HeatRegionDetector` uses `org.opencv:opencv:4.9.0` straight from Maven
  Central (no manual native `.aar`), so no OpenCV SDK download/unzip step
  is required.
- Camera output is shown by baking the overlay into the analyzed frame and
  posting it to a full-screen `ImageView` (matches the original's
  "already-composited frame" design) rather than a passthrough
  `PreviewView` — this keeps behavior identical but costs a bit more
  power than a true preview + overlay-canvas approach.
- Every 2nd frame runs the full pipeline; frames in between call
  `recompose()`/`recomposite()` to re-blend cached masks cheaply, mirroring
  the frame-skip note in the original `heat_detector.py`.
- YUV→BGR conversion assumes `YUV_420_888`, which is what CameraX's
  `ImageAnalysis` always delivers — should not need adjustment.
