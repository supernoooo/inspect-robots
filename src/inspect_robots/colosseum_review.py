"""Review native Colosseum recordings locally before explicit publication.

Adapted from colosseum-client/local/review_evals.py: use Inspect's own resume/import
contracts, support exported videos, and keep preview and scoring separate from upload.
"""

from __future__ import annotations

import html
import json
import math
import secrets
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import parse_qs, unquote, urlsplit

from inspect_robots.colosseum import _CAMERAS, safe_id

_BRIDGES = {"inspect-robots-v1", "inspect-robots-local-v1"}
_CSS = """
:root{color-scheme:light dark;font:15px/1.5 system-ui,sans-serif}
body{margin:0;background:light-dark(#f5f5f2,#171719)}
main{max-width:1400px;margin:auto;padding:20px}h1{font-size:23px}h2{font-size:18px}
.card{border:1px solid #8885;border-radius:10px;padding:16px;margin:14px 0}
a{color:light-dark(#2466bd,#8bb8ff)}table{border-collapse:collapse;width:100%}
td,th{padding:8px;text-align:left;border-bottom:1px solid #8884}
.vids{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
video{width:100%;background:#000;border-radius:5px}.mut{opacity:.7}
.ctl{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;align-items:center}
label{display:block;margin:10px 0 4px}input,select,textarea,button{font:inherit;padding:8px}
textarea{width:100%;box-sizing:border-box;min-height:90px}
button{cursor:pointer;border-radius:5px;border:1px solid #8887}
.submit{background:#b84826;color:white}pre{white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:800px){.vids{grid-template-columns:1fr}}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _page(title: str, body: str) -> bytes:
    return (
        '<!doctype html><html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_escape(title)}</title><style>{_CSS}</style><main>{body}</main></html>"
    ).encode()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _publication_started(local: dict[str, Any]) -> bool:
    return bool(
        local.get("submitted")
        or local.get("remote_evaluation_id")
        or local.get("evaluation_create_pending")
        or local.get("remote_datasets")
        or local.get("datasets_uploaded")
    )


def _legacy_score_0_4(score_0_100: int) -> int:
    """Map an integer percentage to the Router's legacy dataset score, half up."""
    return min(4, (score_0_100 * 4 + 50) // 100)


class ReviewStore:
    """Own local review state; only explicit submission/discard starts a client job."""

    def __init__(self, config_path: str):
        from colosseum_client.robot_config import RobotClientConfig

        self.config_path = Path(config_path).expanduser().resolve()
        self.config = RobotClientConfig.from_yaml(str(self.config_path))
        self.root = Path(self.config.evaluation_dir).expanduser().resolve()
        self.cache = self.root / ".review-cache"
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.preview_lock = threading.Lock()
        self.jobs: dict[str, subprocess.Popen[bytes]] = {}

    def path(self, evaluation_id: str, *parts: str) -> Path:
        """Confine local evidence access to the configured evaluation root."""
        folder = self.root / safe_id(evaluation_id)
        path = folder.joinpath(*parts)
        if not path.resolve().is_relative_to(folder.resolve()) or folder.is_symlink():
            raise ValueError("Evidence path escapes its evaluation folder")
        return path

    def load(self, evaluation_id: str) -> dict[str, Any]:
        """Load only manifests produced by the native Inspect bridge."""
        local = _read(self.path(evaluation_id, "manifest.json"))
        if local.get("bridge") not in _BRIDGES:
            raise ValueError("Evaluation was not created by the Inspect bridge")
        assignment = local["assignment"]
        if assignment["id"] != evaluation_id:
            raise ValueError("Evaluation ID differs from its folder")
        for run in assignment["runs"]:
            safe_id(run["id"])
            if run["side"] not in {"A", "B", "trial"}:
                raise ValueError("Invalid evaluation side")
        return local

    def busy(self, evaluation_id: str) -> bool:
        """Return whether this evaluation has an active upload/discard job."""
        job = self.jobs.get(evaluation_id)
        return job is not None and job.poll() is None

    def complete(self, evaluation_id: str, local: dict[str, Any]) -> bool:
        """Check native completion independently of whether scores were exported."""
        for run in local["assignment"]["runs"]:
            folder = self.path(evaluation_id, run["id"])
            marker = _read(folder / "bridge-trial.json")
            status = _read(folder / "recording-status.json")
            finish = (
                "local_finished"
                if local["bridge"] == "inspect-robots-local-v1"
                else "router_finished"
            )
            if (
                run["state"] != "finished"
                or not marker.get("native_complete")
                or not marker.get(finish)
                or not status.get("complete")
                or not (folder / "frames.jsonl").is_file()
            ):
                return False
        return bool(local["assignment"]["runs"])

    def save(self, evaluation_id: str, form: dict[str, list[str]]) -> None:
        """Save scores/drafts locally, without exporting data or contacting Router."""
        from colosseum_client.evaluation import save

        with self.lock:
            local = self.load(evaluation_id)
            if self.busy(evaluation_id) or local.get("aborted") or _publication_started(local):
                raise ValueError("Scoring is locked after discard or publication starts")
            local_mode = local["bridge"] == "inspect-robots-local-v1"
            outcomes = dict(local.get("outcomes", {}))
            for run in local["assignment"]["runs"]:
                side = run["side"]
                raw = form.get(f"score_{side}", [""])[0].strip()
                if not raw:
                    continue
                number = float(raw)
                limit = 100
                if not math.isfinite(number) or number != int(number) or not 0 <= number <= limit:
                    raise ValueError(f"{side} score must be an integer from 0 to {limit}")
                score = int(number)
                partial = score / limit
                outcomes[side] = {"success": partial == 1, "partial_success": partial}
                if local_mode:
                    # Local dataset import still has a published 0-4 integer contract.
                    # Keep the exact review score in partial_success and derive only
                    # that compatibility field.
                    outcomes[side]["score_0_4"] = _legacy_score_0_4(score)
            preference = form.get("preference", [""])[0]
            feedback = form.get("feedback", [""])[0].strip()
            confidence = form.get("confidence", [""])[0]
            if preference not in {"", "A", "B", "tie"}:
                raise ValueError("Invalid preference")
            if confidence not in {"", "clear", "close", "uncertain"}:
                raise ValueError("Invalid confidence")
            draft: dict[str, Any] = {"preference": preference, "feedback": feedback}
            if local_mode:
                draft["confidence"] = confidence
            open_track = local["assignment"]["track"] == "open"
            ready = (
                all(run["side"] in outcomes for run in local["assignment"]["runs"])
                and (not open_track or bool(preference))
                and (not local_mode or bool(confidence))
            )
            local.update(outcomes=outcomes, draft=draft, review="pending", submission_deferred=True)
            if ready:
                local["result"] = {"outcomes": outcomes, **draft}
                if not open_track:
                    local["result"]["preference"] = None
            else:
                local.pop("result", None)
            save(self.path(evaluation_id, "manifest.json"), local)
            if local.get("excel_path"):
                from inspect_robots.colosseum import sync_excel

                sync_excel(self.path(evaluation_id, "manifest.json"), local)

    def start_job(self, evaluation_id: str, action: str, reason: str = "") -> None:
        """Invoke Inspect submit-only/abort; never resume robot execution from the UI."""
        with self.lock:
            local = self.load(evaluation_id)
            if any(self.busy(key) for key in self.jobs):
                raise ValueError("Another submission/discard is running")
            if local.get("aborted"):
                raise ValueError("Evaluation was discarded")
            argv = [
                sys.executable,
                "-m",
                "inspect_robots.cli",
                "colosseum",
                "--client-config",
                str(self.config_path),
            ]
            if action == "submit":
                if not local.get("result") or not self.complete(evaluation_id, local):
                    raise ValueError("Finish both recordings and save all scores/preference first")
                from colosseum_client.robot_config import RobotClientConfig

                current = RobotClientConfig.from_yaml(str(self.config_path))
                if Path(current.evaluation_dir).expanduser().resolve() != self.root:
                    raise ValueError("Evaluation directory changed; restart the review server")
                if current.skip_upload:
                    raise ValueError(
                        "Set skip_upload: false in the client config to upload all data"
                    )
                argv.extend(["--submit-only", "--resume", evaluation_id])
            elif action == "discard":
                if not reason.strip() or _publication_started(local):
                    raise ValueError("Discard needs a reason and an unpublished evaluation")
                argv.extend(["--abort", evaluation_id, "--reason", reason.strip()])
            else:
                raise ValueError("Unknown review action")
            log_path = self.path(evaluation_id, "review-upload.log")
            log = log_path.open("ab")
            try:
                job = subprocess.Popen(
                    argv,
                    cwd=Path.cwd(),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                log.close()
                raise
            self.jobs[evaluation_id] = job
            threading.Thread(target=self._finish_job, args=(job, log), daemon=True).start()

    @staticmethod
    def _finish_job(job: subprocess.Popen[bytes], log: BinaryIO) -> None:
        code = job.wait()
        log.write(f"\n[exit {code}]\n".encode())
        log.close()

    def preview(self, evaluation_id: str, run_id: str, camera: str) -> Path | None:
        """Encode a wall-clock preview from PNGs, or reuse a nominal-FPS export."""
        local = self.load(evaluation_id)
        if camera not in _CAMERAS or run_id not in {r["id"] for r in local["assignment"]["runs"]}:
            raise ValueError("Unknown run or camera")
        source = self.path(evaluation_id, run_id)
        export = self.path(
            evaluation_id,
            run_id,
            "lerobot",
            "videos",
            f"observation.images.{camera}",
            "chunk-000",
            "file-000.mp4",
        )
        destination = self.cache / evaluation_id / run_id / f"{camera}.mp4"
        if not destination.resolve().is_relative_to(self.root):
            raise ValueError("Preview cache escapes the evaluation root")
        with self.preview_lock:
            if destination.is_file():
                return destination
            frames = source / "frames.jsonl"
            if not frames.is_file():
                return None
            rows = [json.loads(line) for line in frames.read_text().splitlines() if line.strip()]
            if not rows:
                return None
            images = [
                self.path(evaluation_id, run_id, camera, f"{int(row['frame']):06d}.png")
                for row in rows
            ]
            if not all(image.is_file() for image in images):
                return export if export.is_file() else None
            destination.parent.mkdir(parents=True, exist_ok=True)
            fps = _read(source / "recording-context.json").get("fps", 30)
            lines = []
            for index, (row, image) in enumerate(zip(rows, images, strict=True)):
                duration = (
                    rows[index + 1]["seconds"] - row["seconds"]
                    if index + 1 < len(rows)
                    else 1 / fps
                )
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Invalid recording capture timestamps")
                escaped = str(image.resolve()).replace("'", "'\\''")
                lines.extend([f"file '{escaped}'", f"duration {duration:.6f}"])
            lines.append(lines[-2])
            listing = destination.with_suffix(".txt")
            listing.write_text("\n".join(lines) + "\n")
            temporary = destination.with_suffix(".pending.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-vsync",
                    "vfr",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=True,
                timeout=600,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            temporary.replace(destination)
            return destination

    def index(self) -> bytes:
        """Render existing local and Router-assigned evaluations."""
        rows = []
        for path in sorted(
            self.root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                local = self.load(path.parent.name)
            except (ValueError, KeyError):
                continue
            assignment = local["assignment"]
            state = (
                "discarded"
                if local.get("aborted")
                else "submitted"
                if local.get("submitted")
                else "upload in progress / retry"
                if _publication_started(local)
                else "scored; awaiting upload"
                if local.get("result")
                else "needs review"
            )
            if self.busy(assignment["id"]):
                state = "working; refresh for status"
            rows.append(
                f'<tr><td><a href="/ev/{assignment["id"]}">{assignment["id"]}</a></td>'
                f"<td>{_escape(assignment['task']['instruction'])}</td>"
                f"<td>{'complete' if self.complete(assignment['id'], local) else 'incomplete'}</td>"
                f"<td>{state}</td></tr>"
            )
        return _page(
            "LLM Evaluation Review",
            "<h1>LLM Evaluation Review</h1>"
            "<p>Review recordings and save scores locally. "
            "Upload starts only when you click Submit.</p>"
            f'<p class="mut">{_escape(self.root)}</p><div class="card"><table>'
            "<tr><th>Evaluation</th><th>Instruction</th><th>Recordings</th><th>Status</th></tr>"
            + ("".join(rows) or '<tr><td colspan="4">No native evaluations found.</td></tr>')
            + "</table></div>",
        )

    def evaluation_page(self, evaluation_id: str, message: str = "") -> bytes:
        """Render three-camera playback, execution provenance and editable scoring."""
        local = self.load(evaluation_id)
        assignment = local["assignment"]
        local_mode = local["bridge"] == "inspect-robots-local-v1"
        locked = self.busy(evaluation_id) or local.get("aborted") or _publication_started(local)
        disabled = " disabled" if locked else ""
        result = local.get("result", local.get("draft", {}))
        token = f'<input type="hidden" name="token" value="{self.token}">'
        parts = ['<p><a href="/">← All evaluations</a></p>', f"<h1>{evaluation_id}</h1>"]
        if message:
            parts.append(f'<div class="card">{_escape(message)}</div>')
        if locked:
            parts.append(
                "<p>Scoring locked: discarded, uploading or already published. "
                "Refresh for status.</p>"
            )
        parts.append(
            f'<div class="card"><b>Instruction:</b> {_escape(assignment["task"]["instruction"])}'
            f"<br><b>Scene:</b> {_escape(assignment.get('scene', ''))}</div>"
            f'<form method="post">{token}'
        )
        for run in assignment["runs"]:
            side, run_id = run["side"], run["id"]
            folder = self.path(evaluation_id, run_id)
            context = _read(folder / "recording-context.json")
            inspect_log = _read(folder / "inspect-result.json")
            evaluation = inspect_log.get("eval", {})
            policy = evaluation.get("policy_config", {})
            model = run.get("model", {}).get("name", policy.get("model", "unknown"))
            parts.append(f'<div class="card"><h2>{side}: {_escape(model)}</h2>')
            parts.append(
                '<div class="ctl">'
                f"<button type=\"button\" onclick=\"sync('{side}','play')\">Play all</button>"
                f"<button type=\"button\" onclick=\"sync('{side}','pause')\">Pause all</button>"
                f"<button type=\"button\" onclick=\"sync('{side}','restart')\">Restart</button>"
                f"<label>Speed <select onchange=\"rate('{side}',this.value)\">"
                "<option>0.5</option><option selected>1</option>"
                "<option>2</option><option>4</option>"
                '</select></label></div><div class="vids">'
            )
            for camera in context.get("camera_map", _CAMERAS):
                if camera in _CAMERAS:
                    parts.append(
                        f'<div><div>{camera}</div><video data-side="{side}" '
                        f'src="/video/{evaluation_id}/{run_id}/{camera}" controls muted '
                        'preload="none"></video></div>'
                    )
            parts.append(
                '</div><p class="mut">Raw previews preserve capture timing. Existing exported '
                "videos use the nominal FPS of the dataset.</p>"
            )
            summary = {
                "wire": policy.get("wire"),
                "effort": policy.get("effort"),
                "control_hz": evaluation.get("embodiment_info", {}).get("control_hz"),
                "max_llm_calls": policy.get("max_llm_calls"),
                "max_speed_frac": policy.get("max_speed_frac"),
                "max_steps": evaluation.get("max_steps"),
                "steps_executed": inspect_log.get("stats", {}).get("total_steps"),
                "duration_s": inspect_log.get("stats", {}).get("duration_s"),
                "native_control": context.get("native_control_mode"),
                "status": inspect_log.get("status"),
                "error": inspect_log.get("error"),
            }
            parts.append(
                "<details><summary>Execution settings and outcome</summary><pre>"
                + _escape(json.dumps(summary, indent=2, ensure_ascii=False))
                + "</pre>"
            )
            execution = context.get("native_execution")
            if execution:
                parts.append(
                    "<pre>"
                    + _escape(json.dumps(execution, indent=2, ensure_ascii=False))
                    + "</pre>"
                )
            parts.append("</details>")
            outcome = local.get("outcomes", {}).get(side, {})
            value = (
                round(outcome["partial_success"] * 100)
                if outcome
                else ""
            )
            maximum = 100
            parts.append(
                f'<label>{side} score (0-{maximum})</label><input type="number" '
                f'min="0" max="{maximum}" step="1" name="score_{side}" '
                f'value="{value}"{disabled}></div>'
            )
        parts.append('<div class="card">')
        if assignment["track"] == "open":
            parts.append(
                "<label>Preference</label>"
                + self._select(
                    "preference", ("", "A", "B", "tie"), result.get("preference"), disabled
                )
            )
        if local_mode:
            parts.append(
                "<label>Confidence</label>"
                + self._select(
                    "confidence",
                    ("", "clear", "close", "uncertain"),
                    result.get("confidence"),
                    disabled,
                )
            )
        parts.append(
            f'<label>Feedback</label><textarea name="feedback"{disabled}>'
            f"{_escape(result.get('feedback', ''))}</textarea>"
        )
        if not locked:
            parts.append(
                '<div class="ctl"><button name="action" value="save">Save locally</button>'
                '<button class="submit" name="action" value="submit" onclick="return confirm('
                "'Submit scores, feedback, videos and datasets to Colosseum?')\">"
                "Save + Submit all data</button></div>"
            )
        parts.append("</div></form>")
        if (
            _publication_started(local)
            and not local.get("aborted")
            and not self.busy(evaluation_id)
        ):
            parts.append(
                f'<form method="post">{token}<button name="action" value="retry">'
                "Retry / finish upload</button></form>"
            )
        if not locked:
            parts.append(
                f'<form method="post" class="card">{token}<label>Discard reason</label>'
                '<input name="reason" required><button name="action" value="discard" '
                "onclick=\"return confirm('Discard this evaluation? "
                "Local evidence is retained.')\">"
                "Discard</button></form>"
            )
        log = self.path(evaluation_id, "review-upload.log")
        if log.is_file():
            parts.append(
                "<details open><summary>Upload log</summary><pre>"
                + _escape(log.read_text(errors="replace")[-8000:])
                + "</pre></details>"
            )
        parts.append("""<script>
const videos=s=>[...document.querySelectorAll(`video[data-side="${s}"]`)];
function sync(s,a){const v=videos(s),t=v[0]?.currentTime||0;v.forEach(x=>{
if(a==='pause')x.pause();else{x.currentTime=a==='restart'?0:t;x.play().catch(()=>{})}})}
function rate(s,r){videos(s).forEach(v=>v.playbackRate=+r)}
</script>""")
        return _page(evaluation_id, "".join(parts))

    @staticmethod
    def _select(name: str, values: tuple[str, ...], selected: Any, disabled: str) -> str:
        options = "".join(
            f'<option value="{value}"'
            f"{' selected' if selected == value else ''}>{value or '—'}</option>"
            for value in values
        )
        return f'<select name="{name}"{disabled}>{options}</select>'


def handler(store: ReviewStore) -> type[BaseHTTPRequestHandler]:
    """Build a localhost-only handler with protected mutations and seekable videos."""

    class Handler(BaseHTTPRequestHandler):
        def send(
            self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8"
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def route(self) -> list[str]:
            host = self.headers.get("Host", "")
            port = cast(ThreadingHTTPServer, self.server).server_port
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if host not in allowed:
                raise ValueError("Localhost access required")
            return [
                unquote(part) for part in urlsplit(self.path).path.strip("/").split("/") if part
            ]

        def do_GET(self) -> None:
            try:
                route = self.route()
                if not route:
                    self.send(200, store.index())
                elif route[0] == "ev" and len(route) == 2:
                    self.send(200, store.evaluation_page(route[1]))
                elif route[0] == "video" and len(route) == 4:
                    video = store.preview(*route[1:])
                    if video is None:
                        self.send(404, b"No recorded video", "text/plain")
                    else:
                        self.file(video)
                else:
                    self.send(404, b"Not found", "text/plain")
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.send(400, _page("Review error", f"<pre>{_escape(exc)}</pre>"))
            except (subprocess.SubprocessError, OSError):
                self.send(
                    500,
                    b"Preview unavailable; check ffmpeg and local recording files",
                    "text/plain",
                )

        def file(self, path: Path) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            request_range = self.headers.get("Range", "")
            if request_range:
                try:
                    if not request_range.startswith("bytes=") or "," in request_range:
                        raise ValueError("Invalid range")
                    first, separator, last = request_range[6:].partition("-")
                    if not separator:
                        raise ValueError("Invalid range")
                    if first:
                        start, end = int(first), min(int(last), end) if last else end
                    else:
                        start = max(0, size - int(last))
                    if not 0 <= start <= end < size:
                        raise ValueError("Unsatisfiable range")
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if request_range else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if request_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                left = end - start + 1
                while left:
                    chunk = source.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    left -= len(chunk)

        def do_POST(self) -> None:
            try:
                route = self.route()
                if len(route) != 2 or route[0] != "ev":
                    raise ValueError("Invalid review route")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid form size")
                form = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
                if not secrets.compare_digest(form.get("token", [""])[0], store.token):
                    self.send(403, b"Invalid review token", "text/plain")
                    return
                action = form.get("action", ["save"])[0]
                evaluation_id = route[1]
                with store.lock:
                    if action in {"save", "submit"}:
                        store.save(evaluation_id, form)
                        message = "Saved locally. No uploads started."
                        if action == "submit":
                            store.start_job(evaluation_id, "submit")
                            message = "Submission started. Refresh to see the upload log."
                    elif action == "retry":
                        store.start_job(evaluation_id, "submit")
                        message = "Retry started. Refresh to see the upload log."
                    elif action == "discard":
                        store.start_job(evaluation_id, "discard", form.get("reason", [""])[0])
                        message = "Discard started. Local evidence is retained."
                    else:
                        raise ValueError("Unknown review action")
                self.send(200, store.evaluation_page(evaluation_id, message))
            except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
                self.send(400, _page("Review error", f"<pre>{_escape(exc)}</pre>"))

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def serve(config_path: str, port: int = 8767, *, open_browser: bool = False) -> int:
    """Run the local review server without initializing a policy, robot or Router."""
    if not 1 <= port <= 65535:
        raise ValueError("Review port must be from 1 to 65535")
    store = ReviewStore(config_path)
    with ThreadingHTTPServer(("127.0.0.1", port), handler(store)) as server:
        url = f"http://127.0.0.1:{port}"
        print(f"Reviewing {store.root}\nOpen {url} (local only; Ctrl-C to stop)", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nReview server stopped. Local scores and recordings retained.")
    return 0
