#!/usr/bin/env python3
"""Check an assembled Wedeo static release against a deployment path baseline."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--site', required=True, type=Path, help='assembled static site directory')
parser.add_argument('--baseline', required=True, type=Path, help='fresh deployment JSON with a files mapping')
args = parser.parse_args()
root = args.site.resolve()
baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
files = baseline.get('files', baseline)
if not root.is_dir() or not isinstance(files, dict) or not files:
    raise SystemExit('Expected an assembled site and a nonempty deployment files mapping')
if baseline.get('uses_functions'):
    raise SystemExit('This static check cannot validate a deployment with Functions')
paths = set()
for path in root.rglob('*'):
    if path.is_symlink():
        raise SystemExit('Refusing symlink: ' + str(path))
    if path.is_file() and path.relative_to(root).as_posix() != '_headers':
        paths.add('/' + path.relative_to(root).as_posix())
missing = sorted(set(files) - paths)
if missing:
    raise SystemExit('Refusing to omit baseline paths: ' + ', '.join(missing))
if not (root / '_headers').is_file():
    raise SystemExit('Missing _headers release input')

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])

page = root / 'simple/wedeo/index.html'
if not page.is_file():
    raise SystemExit('Missing Wedeo package index')
parsed = Links()
parsed.feed(page.read_text(encoding='utf-8'))
if not parsed.links:
    raise SystemExit('Package index contains no download links')
for href in parsed.links:
    url = urlsplit(href)
    if url.scheme or url.netloc or not url.path.startswith('/downloads/') or not re.fullmatch(r'sha256=[a-f0-9]{64}', url.fragment):
        raise SystemExit('Unexpected package link: ' + href)
    artifact = (root / unquote(url.path).lstrip('/')).resolve()
    if not artifact.is_relative_to((root / 'downloads').resolve()) or not artifact.is_file():
        raise SystemExit('Package download missing: ' + href)
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != url.fragment.removeprefix('sha256='):
        raise SystemExit('Package checksum mismatch: ' + href)
print(json.dumps({'status': 'PASS', 'baseline_paths': len(files), 'package_downloads': len(parsed.links),
                  'extra_paths': sorted(paths - set(files)),
                  'limits': 'Paths and linked package hashes only; compare other asset bytes, headers and routing separately.'}))
