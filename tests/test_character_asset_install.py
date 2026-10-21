"""Exercise asset installation with isolated files, without engine or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location('asset_install',Path(__file__).resolve().parents[1]/'tools/install_character_assets.py')
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)

def bundle(tmp_path):
    raw = b'asset contents'
    manifest = {'bundle_id':'test-binding','files':[{'path':'model.bin','bytes':len(raw),
                'sha256':hashlib.sha256(raw).hexdigest()}]}
    archive = tmp_path/'assets.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('Character/model.bin',raw)
        z.writestr('Character/ASSET_BUNDLE.json',json.dumps(manifest))
    descriptor = {'bundle_id':'test-binding','root':'Character','archive_name':archive.name,
                  'manifest_sha256':hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
                  'archive_bytes':archive.stat().st_size,'sha256':installer.sha256(archive),'download_url':None}
    return archive, descriptor

def test_install_reuse_and_preserve_modified_asset(tmp_path):
    archive, descriptor = bundle(tmp_path)
    target = tmp_path/'installed'
    assert installer.install(descriptor,target,archive)['status'] == 'installed'
    assert (target/'model.bin').read_bytes() == b'asset contents'
    assert installer.install(descriptor,target,archive)['status'] == 'already_installed'
    (target/'model.bin').write_bytes(b'user edit')
    with pytest.raises(ValueError,match='hash mismatch'):
        installer.install(descriptor,target,archive)
    assert (target/'model.bin').read_bytes() == b'user edit'
    (target/'ASSET_BUNDLE.json').write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='manifest hash mismatch'):
        installer.install(descriptor,target,archive)

def test_corrupt_archive_leaves_no_install(tmp_path):
    archive, descriptor = bundle(tmp_path)
    archive.write_bytes(archive.read_bytes()+b'corruption')
    target = tmp_path/'installed'
    with pytest.raises(ValueError,match='SHA-256'):
        installer.install(descriptor,target,archive)
    assert not target.exists()

def test_archive_cannot_write_outside_bundle(tmp_path):
    archive, descriptor = bundle(tmp_path)
    with zipfile.ZipFile(archive,'a') as z:
        z.writestr('../outside.txt',b'outside')
    descriptor.update(archive_bytes=archive.stat().st_size,sha256=installer.sha256(archive))
    with pytest.raises(ValueError,match='outside'):
        installer.install(descriptor,tmp_path/'installed',archive)
    assert not (tmp_path/'outside.txt').exists()
    assert not (tmp_path/'installed').exists()
