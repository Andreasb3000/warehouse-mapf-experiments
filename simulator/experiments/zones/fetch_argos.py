#!/usr/bin/env python3
"""Download the official, checksum-pinned ARGoS beta59 development package."""
import argparse
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SHA256 = "2bcd19430266e80a54a053d9b5079212468f5f518042921eadb7cbd2efa71fd8"
FILE_ID = "1oO2lb2LuLq4IrZmNMiJurWTotHp_pDye"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs.get("value", "")


def fetch(output):
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == SHA256:
        print(f"Verified {output}")
        return
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if not data.startswith(b"!<arch>\n"):
        parser = DownloadForm()
        parser.feed(data.decode("utf-8"))
        if parser.fields.get("id") != FILE_ID:
            raise RuntimeError("Official ARGoS download is unavailable; no output written")
        url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(parser.fields)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"ARGoS package checksum mismatch: {digest}; no output written")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f"Downloaded and verified {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/argos3-beta59-ubuntu20.deb")
    fetch(parser.parse_args().output)
