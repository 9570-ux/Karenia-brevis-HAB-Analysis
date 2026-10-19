"""Download the verified NOAA archive bytes, using validated HTTP ranges.

Ranges are transport only: bytes are assembled in their original order, with
no CSV rewriting, scientific transformation, or data from other sources.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv"
EXPECTED_SIZE = 29951499  # Verified NOAA FTP listing and HTTPS Content-Length.
CHUNK = 262144


def main():
    destination = ROOT / "data/raw/habsos_20240430.csv"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing raw data")
    staging = ROOT / "docs/download_staging"
    staging.mkdir(exist_ok=True)
    request = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
        modified = response.headers.get("Last-Modified")
    if size != EXPECTED_SIZE:
        raise ValueError("NOAA file size changed; re-inspect metadata before downloading")

    def fetch(start):
        end = min(start + CHUNK, size) - 1
        part = staging / f"{start:09d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(3):
            try:
                headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
                if etag:
                    headers["If-Match"] = etag
                req = urllib.request.Request(URL, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("NOAA did not return the requested byte range")
                    if etag and response.headers.get("ETag") != etag:
                        raise ValueError("NOAA file changed during download")
                    payload = response.read()
                if len(payload) != end - start + 1:
                    raise ValueError("Incomplete byte range")
                part.write_bytes(payload)
                return part
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)

    starts = list(range(0, size, CHUNK))
    with ThreadPoolExecutor(max_workers=16) as pool:
        pending = [pool.submit(fetch, start) for start in starts]
        for completed, task in enumerate(as_completed(pending), 1):
            task.result()
            if completed % 10 == 0 or completed == len(starts):
                print(f"Verified byte ranges: {completed}/{len(starts)}", flush=True)
    assembled = staging / "complete.csv"
    with assembled.open("wb") as output:
        for start in starts:
            output.write((staging / f"{start:09d}.part").read_bytes())
    if assembled.stat().st_size != size:
        raise ValueError("Assembled file size mismatch")
    # Verify CSV readability before designating it as the original raw file.
    from inspect_dataset import inspect
    report = inspect(assembled)
    if report["row_count"] <= 0:
        raise ValueError("Downloaded CSV has no records")
    assembled.rename(destination)
    manifest = {"source_url": URL, "accession": "0120767", "version": "8.8",
                "filename": destination.name, "size_bytes": size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "etag": etag, "last_modified": modified, "accessed_date": "2026-10-14",
                "transport": "HTTP byte ranges validated and assembled without rewriting CSV"}
    (ROOT / "docs/download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
