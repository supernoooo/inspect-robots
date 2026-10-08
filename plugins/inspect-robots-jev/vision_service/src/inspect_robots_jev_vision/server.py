"""Small JSON/HTTP vision process; only RGB and request metadata cross the boundary."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from inspect_robots_jev.vision_protocol import (
    VERSION, PROFILE_VERSION, PROMPT_VERSIONS, SERVICE_VERSION, VisionProtocolError,
    decode_request, decode_response, dumps, encode_mask, loads, model_version,
)

from .backend import Backend, PretrainedBackend

MAX_REQUEST_BYTES = 16_000_000


def make_handler(backend: Backend) -> type[BaseHTTPRequestHandler]:
    model = model_version(backend.model_version)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict[str, Any]) -> None:
            payload = dumps(body)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.path != "/v1/detect":
                self._send(404, {"version": VERSION, "error": "not_found"})
                return
            length = self.headers.get("Content-Length", "")
            if not length.isdecimal() or int(length) > MAX_REQUEST_BYTES:
                self._send(413, {"version": VERSION, "error": "request_too_large"})
                return
            response_version = VERSION
            try:
                image, camera, request_id, captured_at, version, profile = decode_request(
                    loads(self.rfile.read(int(length))))
                response_version = version
                height, width = image.shape[:2]
                fingerprint = model if version == VERSION else {
                    **model, "prompt_version": PROMPT_VERSIONS[profile],
                    "service_version": SERVICE_VERSION}
                rows = []
                regions = backend.infer(image) if version == VERSION else backend.infer(image, profile)
                for region in regions:
                    rows.append({"category": region.category, "box": list(region.box),
                                 "mask": encode_mask(region.mask),
                                 "detection_score": region.detection_score,
                                 "segmentation_score": region.segmentation_score,
                                 "occluded": region.occluded, "camera": camera,
                                 "request_id": request_id, "captured_at": captured_at,
                                 "model_version": fingerprint})
                body = {"version": version, "request_id": request_id,
                        "camera": camera, "captured_at": captured_at,
                        "height": height, "width": width,
                        "model_version": fingerprint, "detections": rows}
                if version == PROFILE_VERSION:
                    body["task_profile"] = profile
                decode_response(body, request_id=request_id, camera=camera,
                                captured_at=captured_at, height=height, width=width,
                                task_profile=profile if version == PROFILE_VERSION else None)
                self._send(200, body)
            except VisionProtocolError as exc:
                self._send(400, {"version": response_version, "error": exc.code})
            except Exception:
                self._send(500, {"version": VERSION, "error": "inference_error"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Jev RGB vision service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--detector-revision", default="main")
    parser.add_argument("--segmenter-revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    backend = PretrainedBackend(detector_revision=args.detector_revision,
                                segmenter_revision=args.segmenter_revision,
                                local_files_only=args.local_files_only, device=args.device)
    print({**backend.model_version, "prompt_versions": PROMPT_VERSIONS,
           "service_version": SERVICE_VERSION}, flush=True)
    with HTTPServer((args.host, args.port), make_handler(backend)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
