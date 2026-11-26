#!/usr/bin/env python3
"""Check an assembled Wedeo static release against a deployment path baseline."""
import argparse
import hashlib
from email.parser import BytesParser
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tarfile
from zipfile import ZipFile
from urllib.parse import unquote, urljoin, urlsplit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--site', required=True, type=Path, help='assembled static site directory')
parser.add_argument('--baseline', required=True, type=Path, help='fresh deployment JSON with a files mapping')
parser.add_argument('--reference-index', required=True, type=Path, help='current production package-index HTML')
parser.add_argument('--allow-index-removal', action='store_true', help='allow explicitly reviewed changes to existing package links')
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
            self.links.append(attrs)

root_index = root / 'simple/index.html'
if not root_index.is_file():
    raise SystemExit('Missing package index root')
root_links = Links()
root_links.feed(root_index.read_text(encoding='utf-8'))
expected_project = 'https://release.invalid/simple/wedeo/'
if not any(urljoin('https://release.invalid/simple/', link['href']) == expected_project for link in root_links.links):
    raise SystemExit('Package index root does not link to Wedeo')

page = root / 'simple/wedeo/index.html'
if not page.is_file():
    raise SystemExit('Missing Wedeo package index')
parsed = Links()
parsed.feed(page.read_text(encoding='utf-8'))
if not parsed.links:
    raise SystemExit('Package index contains no download links')
reference = Links()
reference.feed(args.reference_index.read_text(encoding='utf-8'))
if not reference.links:
    raise SystemExit('Reference package index contains no download links')
def signature(link):
    return (link['href'], link.get('data-requires-python', ''))
removed = sorted(set(map(signature, reference.links)) - set(map(signature, parsed.links)))
if removed and not args.allow_index_removal:
    raise SystemExit('Existing package links or requirements removed; review before release: ' + repr(removed))
for link in parsed.links:
    href = link['href']
    url = urlsplit(href)
    if url.scheme or url.netloc or not url.path.startswith('/downloads/') or not re.fullmatch(r'sha256=[a-f0-9]{64}', url.fragment):
        raise SystemExit('Unexpected package link: ' + href)
    artifact = (root / unquote(url.path).lstrip('/')).resolve()
    if not artifact.is_relative_to((root / 'downloads').resolve()) or not artifact.is_file():
        raise SystemExit('Package download missing: ' + href)
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != url.fragment.removeprefix('sha256='):
        raise SystemExit('Package checksum mismatch: ' + href)
    if artifact.name.endswith('.whl'):
        with ZipFile(artifact) as archive:
            entries = [info for info in archive.infolist() if info.filename.endswith('.dist-info/METADATA')]
            if len(entries) != 1 or entries[0].file_size > 1024 * 1024:
                raise SystemExit('Unexpected wheel metadata: ' + href)
            metadata = BytesParser().parsebytes(archive.read(entries[0]))
    elif artifact.name.endswith('.tar.gz'):
        with tarfile.open(artifact, 'r:gz') as archive:
            entries = [item for item in archive.getmembers() if item.isfile() and item.name.count('/') == 1 and item.name.endswith('/PKG-INFO')]
            if len(entries) != 1 or entries[0].size > 1024 * 1024:
                raise SystemExit('Unexpected source archive metadata: ' + href)
            metadata = BytesParser().parsebytes(archive.extractfile(entries[0]).read())
    else:
        raise SystemExit('Unsupported package artifact; review verifier: ' + href)
    project = re.sub(r'[-_.]+', '-', str(metadata.get('Name', ''))).lower()
    version = str(metadata.get('Version', '')).strip()
    if project != 'wedeo' or not version:
        raise SystemExit('Unexpected package name or version in metadata: ' + href)
    # Deliberately conservative: unusual but valid version spellings require review.
    if artifact.name.endswith('.whl'):
        components = artifact.name[:-4].split('-')
        filename_matches = (len(components) in (5, 6) and components[0].lower() == 'wedeo'
                            and components[1] == version)
    else:
        filename_matches = artifact.name == f'wedeo-{version}.tar.gz'
    if not filename_matches:
        raise SystemExit('Package filename differs from artifact name/version metadata: ' + href)
    requirement = metadata.get('Requires-Python')
    if not requirement or link.get('data-requires-python') != requirement:
        raise SystemExit('Python requirement differs from artifact metadata: ' + href)
print(json.dumps({'status': 'PASS', 'baseline_paths': len(files), 'package_index_root_link': True, 'package_downloads': len(parsed.links), 'python_requirements_checked': True, 'package_names_and_versions_checked': True,
                  'reviewed_removed_links': removed, 'extra_paths': sorted(paths - set(files)),
                  'limits': 'Paths, linked package hashes, names, versions and Python requirements only; compare other asset bytes, headers and routing separately.'}))
