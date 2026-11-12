"""Offline repository checks: links, CLI examples, asset/data/model integrity.

This checks files and invokes --help only. It never starts a device connection,
installs dependencies, downloads assets, trains, or runs a hardware command.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def headings(text: str) -> set[str]:
    result = set()
    counts = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("#"):
            title = re.sub(r"^#+\s+", "", line).lower()
            slug = re.sub(r"[^\w\- ]", "", title).replace(" ", "-")
            count = counts.get(slug, 0)
            result.add(f"{slug}-{count}" if count else slug)
            counts[slug] = count + 1
    return result


def check_documents() -> int:
    documents = [ROOT / name for name in ("README.md", "README.zh.md", "CONTRIBUTING.md",
                 "CHANGELOG.md", "THIRD_PARTY_NOTICES.md", "data/README.md", "out/README.md")]
    documents += sorted((ROOT / "docs").glob("*.md"))
    for path in documents:
        text = path.read_text()
        if text.count("```") % 2:
            raise ValueError(f"Unbalanced code fences in {path.relative_to(ROOT)}")
        for href in re.findall(r"\]\(([^)]+)\)", text):
            if re.match(r"[a-z][a-z0-9+.-]*:", href, re.I):
                continue
            target, _, fragment = href.partition("#")
            linked = (path.parent / target).resolve() if target else path
            if not linked.exists():
                raise ValueError(f"Missing link in {path.relative_to(ROOT)}: {href}")
            if fragment and linked.suffix == ".md" and fragment not in headings(linked.read_text()):
                raise ValueError(f"Missing heading in {path.relative_to(ROOT)}: {href}")
    return len(documents)


def check_examples() -> int:
    help_cache = {}
    checked = 0
    for name in ("README.md", "README.zh.md"):
        for block in re.findall(r"```bash\n(.*?)```", (ROOT / name).read_text(), re.S):
            for line in block.replace("\\\n", " ").splitlines():
                parts = shlex.split(line, comments=True)
                while parts and re.match(r"[A-Za-z_][A-Za-z0-9_]*=", parts[0]):
                    parts.pop(0)
                if not parts or parts[0] != "python" or len(parts) < 2:
                    continue
                if parts[1] == "-c" or parts[1:3] in (["-m", "pip"], ["-m", "unittest"]):
                    continue
                prefix = parts[1:3] if parts[1] == "-m" else parts[1:2]
                arguments = parts[1 + len(prefix):]
                if prefix[0] != "-m" and not (ROOT / prefix[0]).is_file():
                    raise ValueError(f"Missing documented script: {prefix[0]}")
                key = tuple(prefix)
                if key not in help_cache:
                    result = subprocess.run([sys.executable, *prefix, "--help"], cwd=ROOT,
                                            capture_output=True, text=True, timeout=30)
                    if result.returncode:
                        raise ValueError(f"{key} --help failed: {result.stderr}")
                    help_cache[key] = set(re.findall(r"--[a-z][a-z0-9-]*", result.stdout))
                unknown = {arg for arg in arguments if arg.startswith("--")} - help_cache[key]
                if unknown:
                    raise ValueError(f"Unknown documented options for {key}: {unknown}")
                checked += 1
    return checked


def check_manifests() -> tuple[int, int]:
    artifacts = json.loads((ROOT / "artifacts/manifest.json").read_text())["artifacts"]
    for entry in artifacts:
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Artifact differs from manifest: {entry['path']}")
    components = json.loads((ROOT / "assets/manifest.json").read_text())["components"]
    for entry in components:
        root = ROOT / entry["path"]
        paths = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        h = hashlib.sha256()
        for path in paths:
            h.update(path.relative_to(root).as_posix().encode() + b"\0" + digest(path).encode() + b"\n")
        if len(paths) != entry["files"] or h.hexdigest() != entry["tree_sha256"]:
            raise ValueError(f"Asset tree differs from manifest: {entry['path']}")
        if not (ROOT / entry["notice"]).is_file():
            raise ValueError(f"Missing asset notice: {entry['notice']}")
    return len(artifacts), len(components)


def main() -> None:
    docs = check_documents()
    commands = check_examples()
    artifacts, components = check_manifests()
    print(f"PASS: {docs} documents, {commands} CLI examples, {artifacts} artifacts, {components} asset trees")


if __name__ == "__main__":
    main()
