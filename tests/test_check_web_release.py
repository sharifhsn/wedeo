"""Process-boundary checks for the static Wedeo release verifier."""
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from zipfile import ZipFile

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_web_release.py'

class ReleaseCheckTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wedeo-release-test-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.site = root / 'site'
        (self.site / 'downloads').mkdir(parents=True)
        (self.site / 'simple/wedeo').mkdir(parents=True)
        (self.site / '_headers').write_text('/*\n  Vary: Accept-Encoding\n')
        (self.site / 'simple/index.html').write_text('<a href="wedeo/">wedeo</a>')
        metadata = EmailMessage()
        metadata['Name'] = 'wedeo'
        metadata['Version'] = '0.3.1'
        metadata['Requires-Python'] = '>=3.8'
        body = metadata.as_bytes()
        self.wheel = self.site / 'downloads/wedeo-0.3.1-py3-none-any.whl'
        with ZipFile(self.wheel, 'w') as archive:
            archive.writestr('wedeo-0.3.1.dist-info/METADATA', body)
        self.sdist = self.site / 'downloads/wedeo-0.3.1.tar.gz'
        with tarfile.open(self.sdist, 'w:gz') as archive:
            item = tarfile.TarInfo('wedeo-0.3.1/PKG-INFO')
            item.size = len(body)
            archive.addfile(item, BytesIO(body))
        def anchor(artifact):
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            return f'<a href="/downloads/{artifact.name}#sha256={digest}" data-requires-python="&gt;=3.8">{artifact.name}</a>'
        self.wheel_link = anchor(self.wheel)
        self.sdist_link = anchor(self.sdist)
        self.index = self.site / 'simple/wedeo/index.html'
        self.index.write_text(self.wheel_link + self.sdist_link)
        self.reference = root / 'reference.html'
        self.reference.write_text(self.index.read_text())
        self.baseline = root / 'baseline.json'
        paths = {'/' + p.relative_to(self.site).as_posix(): 'placeholder' for p in self.site.rglob('*') if p.is_file() and p.name != '_headers'}
        self.baseline.write_text(json.dumps({'files': paths, 'uses_functions': False}))

    def run_check(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), '--site', str(self.site),
                               '--baseline', str(self.baseline), '--reference-index', str(self.reference), *extra],
                              capture_output=True, text=True)

    def test_complete_release_passes(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['package_downloads'], 2)

    def test_missing_parent_index_link_fails(self):
        (self.site / 'simple/index.html').write_text('<a href="missing/">missing</a>')
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Package index root does not link to Wedeo', result.stderr)

    def test_missing_baseline_path_fails(self):
        self.wheel.unlink()
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing to omit baseline paths', result.stderr)

    def test_unreviewed_link_removal_fails(self):
        self.index.write_text(self.sdist_link)
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Existing package links or requirements removed', result.stderr)
        reviewed = self.run_check('--allow-index-removal')
        self.assertEqual(reviewed.returncode, 0, reviewed.stderr)
        self.assertEqual(len(json.loads(reviewed.stdout)['reviewed_removed_links']), 1)

    def test_checksum_mismatch_fails(self):
        self.wheel.write_bytes(self.wheel.read_bytes() + b'changed')
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Package checksum mismatch', result.stderr)

    def test_python_requirement_mismatch_fails(self):
        self.index.write_text(self.index.read_text().replace('&gt;=3.8', '&gt;=3.9', 1))
        result = self.run_check('--allow-index-removal')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python requirement differs', result.stderr)

    def test_misleading_version_filename_fails(self):
        renamed = self.wheel.with_name('wedeo-9.9-py3-none-any.whl')
        shutil.copyfile(self.wheel, renamed)
        self.index.write_text(self.index.read_text().replace(self.wheel.name, renamed.name))
        result = self.run_check('--allow-index-removal')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Package filename differs', result.stderr)

if __name__ == '__main__':
    unittest.main()
