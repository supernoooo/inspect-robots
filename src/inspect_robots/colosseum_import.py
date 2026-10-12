"""Publish locally selected LLM pairs through existing manual/import endpoints."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from inspect_robots import colosseum as bridge


class ImportHTTPError(RuntimeError):
    """A definite HTTP rejection, distinguished from a lost response."""

    def __init__(self, status: int, reason: str):
        self.status = status
        super().__init__(f"Local evaluation import HTTP {status}: {reason}")


class LocalUploadAPI:
    """Use Client credentials at Router; keep signed storage requests credential-free."""

    def __init__(self, config: Any):
        from colosseum_client.evaluation import EvalAPI

        self.config = config
        self.api = EvalAPI(config)
        self.client = self.api.client

    def request(self, method: str, path: str, body: Any = None) -> Any:
        """Request an API route without logging request credentials."""
        import httpx
        from colosseum_client.http_diagnostics import log_http_error

        try:
            response = self.client.request(
                method, path, json=body, timeout=1800 if path.endswith("/complete") else 30
            )
        except httpx.HTTPError:
            raise RuntimeError(
                "Local evaluation upload connection failed; retry saved submission"
            ) from None
        if response.is_error:
            reason = log_http_error(
                response, operation="Local evaluation import", log_dir=self.config.evaluation_dir
            )
            raise ImportHTTPError(response.status_code, reason)
        return response.json()

    def put(self, target: dict[str, Any], path: Path, local_route: str | None = None) -> None:
        """Stream a real file to its local or presigned upload target."""
        import httpx
        from colosseum_client.http_diagnostics import log_http_error

        if target.get("complete"):
            return
        try:
            with path.open("rb") as source:
                if target.get("storage") == "local" and local_route:
                    response = self.client.put(
                        local_route,
                        content=source,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=600,
                    )
                elif target.get("url"):
                    # No Router auth/cookies on S3 or other presigned targets.
                    with httpx.Client(timeout=600, trust_env=False) as storage:
                        response = storage.put(
                            target["url"], content=source, headers=target.get("headers", {})
                        )
                else:
                    raise ValueError("Upload target has no supported storage destination")
        except httpx.HTTPError:
            raise RuntimeError("Storage upload connection failed; retry saved submission") from None
        if response.status_code not in {200, 201, 204, 412}:
            reason = log_http_error(
                response, operation="Local dataset/video upload", log_dir=self.config.evaluation_dir
            )
            raise RuntimeError(f"Storage upload HTTP {response.status_code}: {reason}")

    def close(self) -> None:
        """Close the authenticated HTTP connection."""
        self.api.close()


def import_manifest(local: dict[str, Any], run: dict[str, Any], dataset: Path) -> dict[str, Any]:
    """Import real LeRobot evidence with the server's published 0-4 scoring contract."""
    from colosseum_client.evaluation import sha256_stream

    files = []
    for path in sorted(dataset.rglob("*")):
        if path.is_symlink():
            raise ValueError("Dataset import cannot follow symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(dataset).as_posix()
        size = path.stat().st_size
        if not 0 < size <= 536870912 or len(relative) > 300:
            raise ValueError("Dataset file exceeds Router import limits")
        with path.open("rb") as source:
            digest = base64.b64encode(sha256_stream(source)).decode()
        files.append({"path": relative, "size": size, "checksum_sha256": digest})
    if not 5 <= len(files) <= 6000 or not (dataset / "eval_trials.csv").is_file():
        raise ValueError("Dataset lacks the scored LeRobot import files")
    return {
        "submission_id": run["id"],
        "name": f"Inspect {run['model']['name']} {local['assignment']['id']}",
        "robot_id": local["assignment"]["robot_id"],
        "task_id": local["task_id"],
        # API model revisions are not HF commit hashes; preserve them in colosseum.json.
        "model": {"name": run["model"]["name"], "repo_id": "", "revision": ""},
        "scoring": "score-0-4-success-4-v1",
        "results_file": "eval_trials.csv",
        "camera_map": {
            camera: f"observation.images.{camera}"
            for camera in local["assignment"]["task"]["cameras"]
        },
        "files": files,
    }


def _notes(local: dict[str, Any]) -> str:
    return json.dumps(
        {
            "inspect_local_id": local["assignment"]["id"],
            "source": "inspect-robots locally selected LLM pair",
            "models": {run["side"]: run["model"] for run in local["assignment"]["runs"]},
            "datasets": local.get("remote_datasets", {}),
            "execution": local.get("execution", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def evaluation_payload(local: dict[str, Any], folder: Path) -> dict[str, Any]:
    """Build the A/B record from actual saved human scores and identities."""
    assignment = local["assignment"]
    runs = {run["side"]: run for run in assignment["runs"]}
    durations = []
    for run in runs.values():
        log = json.loads((folder / run["id"] / "inspect-result.json").read_text())
        durations.append(float(log.get("stats", {}).get("duration_s", 0) or 0))
    scores = {side: local["outcomes"][side]["partial_success"] for side in ("A", "B")}
    return {
        "robotId": assignment["robot_id"],
        "task": assignment["task"]["instruction"],
        "instruction": assignment["task"]["instruction"],
        "institution": assignment["institution"],
        "scene": assignment["scene"],
        "date": local["created"][:10],
        "evalTime": local["created"],
        "policyA": runs["A"]["policy_id"],
        "policyB": runs["B"]["policy_id"],
        "winner": local["result"]["preference"],
        "partialSuccessA": scores["A"],
        "partialSuccessB": scores["B"],
        "partialSuccess": (scores["A"] + scores["B"]) / 2,
        "duration": f"{sum(durations):.1f}s",
        "evaluator": assignment["evaluator"],
        "feedback": local["result"]["feedback"],
        "confidence": local["result"]["confidence"],
        "notes": _notes(local),
    }


def _find_existing(api: LocalUploadAPI, local: dict[str, Any]) -> str | None:
    robot = quote(local["assignment"]["robot_id"], safe="")
    response = api.request("GET", f"/api/list_ab_evaluations?robot={robot}&track=open")
    rows = (
        response
        if isinstance(response, list)
        else response.get("evaluations", response.get("items", []))
    )
    found = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            notes = json.loads(row.get("notes", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(notes, dict) and notes.get("inspect_local_id") == local["assignment"]["id"]:
            found.append(bridge.safe_id(row["id"]))
    if len(found) > 1:
        raise ValueError("Multiple web records have this local pair ID; resolve before uploading")
    return found[0] if found else None


def publish_local_pair(config: Any, manifest: Path, local: dict[str, Any]) -> None:
    """Publish verified local evidence and finalize the selected comparison."""
    from colosseum_client.evaluation import save

    from inspect_robots.colosseum_local import validate_completed

    folder = manifest.parent
    validate_completed(local, folder, config)
    if config.test:
        raise ValueError(
            "Local test runs cannot be published through manual evaluation APIs "
            "that lack a test flag"
        )
    if config.dataset_url:
        raise ValueError(
            "Local imports use Router dataset storage; remove dataset_url for this mode"
        )
    api = LocalUploadAPI(config)
    try:
        # Stable notes recover a lost POST reply without creating duplicate comparisons.
        if not local.get("remote_evaluation_id"):
            existing = _find_existing(api, local)
            if not existing:
                if local.get("evaluation_create_pending"):
                    raise ValueError(
                        "Previous evaluation creation was unconfirmed and absent from "
                        "web listing; reconcile before retrying"
                    )
                payload = {**evaluation_payload(local, folder), "token": config.token}
                local["evaluation_create_pending"] = True
                save(manifest, local)
                try:
                    response = api.request("POST", "/api/evaluations", payload)
                except ImportHTTPError as exc:
                    if 400 <= exc.status < 500:
                        local.pop("evaluation_create_pending", None)
                        save(manifest, local)
                    raise
                existing = bridge.safe_id(response["id"])
            local["remote_evaluation_id"] = existing
            local.pop("evaluation_create_pending", None)
            save(manifest, local)
        evaluation_id = bridge.safe_id(local["remote_evaluation_id"])
        if not config.skip_upload:
            # Register, checksum/size verify and publish each real run independently.
            for run in local["assignment"]["runs"]:
                run_id = run["id"]
                if local.get("datasets_uploaded", {}).get(run_id):
                    continue
                dataset = folder / run_id / "lerobot"
                description = import_manifest(local, run, dataset)
                dataset_id = local.get("remote_datasets", {}).get(run_id)
                if not dataset_id:
                    response = api.request("POST", "/api/datasets", description)
                    dataset_id = bridge.safe_id(response.get("id", response.get("dataset_id")))
                    local.setdefault("remote_datasets", {})[run_id] = dataset_id
                    save(manifest, local)
                route = "/api/datasets/" + bridge.safe_id(dataset_id)
                for index, file in enumerate(description["files"], 1):
                    query = "?path=" + quote(file["path"], safe="")
                    target = api.request("POST", route + "/upload-url" + query)
                    print(
                        f"Uploading {run['side']} dataset {index}/{len(description['files'])}: "
                        f"{file['path']}"
                    )
                    api.put(target, dataset / file["path"], route + "/file" + query)
                completed = api.request("POST", route + "/complete")
                if completed.get("state") in {"failed", "error"} or completed.get("error"):
                    raise ValueError("Router dataset validation failed; publication stopped")
                published = api.request("POST", route + "/publish")
                if published.get("state") in {"failed", "error"} or published.get("error"):
                    raise ValueError("Router dataset publication failed")
                local.setdefault("datasets_uploaded", {})[run_id] = True
                save(manifest, local)
                bridge.sync_excel(manifest, local)
            if not local.get("videos_uploaded"):
                cameras = {
                    "head_image": "top",
                    "left_image": "left_wrist",
                    "right_image": "right_wrist",
                }
                sources = {
                    (run["side"], cameras[camera]): folder
                    / run["id"]
                    / "lerobot"
                    / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
                    for run in local["assignment"]["runs"]
                    for camera in local["assignment"]["task"]["cameras"]
                }
                targets = api.request(
                    "POST",
                    f"/api/evaluations/{evaluation_id}/videos/upload-urls",
                    {
                        "token": config.token,
                        "videos": [
                            {"side": side, "camera": camera, "contentType": "video/mp4"}
                            for side, camera in sources
                        ],
                    },
                )
                uploads = targets["uploads"]
                if {(target["side"], target["camera"]) for target in uploads} != set(
                    sources
                ) or len(uploads) != len(sources):
                    raise ValueError("Web video upload targets do not cover the recorded cameras")
                for target in uploads:
                    api.put(
                        {
                            "url": target["url"],
                            "headers": {"Content-Type": target.get("contentType", "video/mp4")},
                        },
                        sources[(target["side"], target["camera"])],
                    )
                local["videos_uploaded"] = True
                save(manifest, local)
        result = api.request(
            "PATCH",
            f"/api/evaluations/{evaluation_id}/result",
            {
                "token": config.token,
                "pref": local["result"]["preference"],
                "partialSuccessA": local["outcomes"]["A"]["partial_success"],
                "partialSuccessB": local["outcomes"]["B"]["partial_success"],
                "feedback": local["result"]["feedback"],
                "notes": _notes(local),
                "finalize": True,
            },
        )
        if result.get("finalized") is False:
            raise ValueError("Web evaluation result was not finalized")
        local.update(submitted=True, submission_deferred=False)
        save(manifest, local)
        bridge.sync_excel(manifest, local)
        print(
            f"Web evaluation submitted: {evaluation_id}; source: locally selected Inspect LLM pair."
        )
        if config.skip_upload:
            print("Datasets retained locally; web scores submitted.")
        else:
            print("LLM datasets uploaded, verified and published; A/B review videos uploaded.")
    finally:
        api.close()
