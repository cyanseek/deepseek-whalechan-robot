#!/usr/bin/env python3
"""Install the version-matched character ZIP, from a local file or HTTPS."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT/'assets/characters/bluefatfish-runtime.json'
DESTINATION = ROOT/'assets/local/BlueFatFish_Visual_Delivery'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def verify_delivery(folder, descriptor):
    if sha256(folder/'ASSET_BUNDLE.json') != descriptor['manifest_sha256']:
        raise ValueError('Asset manifest hash mismatch; existing files were preserved.')
    manifest = json.loads((folder/'ASSET_BUNDLE.json').read_text('utf8'))
    if manifest['bundle_id'] != descriptor['bundle_id']:
        raise ValueError('Existing assets belong to another bundle; choose a new destination.')
    for entry in manifest['files']:
        p = (folder/entry['path']).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file():
            raise ValueError('Incomplete asset bundle: '+entry['path'])
        if p.stat().st_size != entry['bytes'] or sha256(p) != entry['sha256']:
            raise ValueError('Asset hash mismatch: '+entry['path'])
    return len(manifest['files'])

def install(descriptor, destination, archive=None, url=None):
    destination = Path(destination).resolve()
    if destination.exists():
        return {'status':'already_installed','delivery':str(destination),
                'verified_files':verify_delivery(destination,descriptor)}
    if archive is None:
        url = url or descriptor.get('download_url')
        if not url:
            raise ValueError('No published download URL is configured yet. Use --archive with '+descriptor['archive_name']+'.')
        if not url.startswith('https://'):
            raise ValueError('Remote asset URLs must use HTTPS.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bluefatfish-install-',dir=destination.parent) as temp:
        temp = Path(temp)
        if archive is None:
            archive = temp/descriptor['archive_name']
            request = urllib.request.Request(url,headers={'User-Agent':'DeepSeek-Whalechan-Robot asset installer'})
            with urllib.request.urlopen(request,timeout=60) as response, archive.open('wb') as output:
                size = 0
                while block := response.read(1024*1024):
                    size += len(block)
                    if size > descriptor['archive_bytes']:
                        raise ValueError('Downloaded archive exceeds the expected size.')
                    output.write(block)
        archive = Path(archive).resolve()
        if archive.stat().st_size != descriptor['archive_bytes'] or sha256(archive) != descriptor['sha256']:
            raise ValueError('Archive size or SHA-256 does not match the current runtime binding.')
        staging = temp/'unpacked'
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                parts = PurePosixPath(member.filename).parts
                target = (staging/member.filename).resolve()
                if not parts or parts[0] != descriptor['root'] or not target.is_relative_to(staging.resolve()):
                    raise ValueError('Archive member is outside the character directory.')
            z.extractall(staging)
        prepared = staging/descriptor['root']
        count = verify_delivery(prepared,descriptor)
        if destination.exists():
            raise ValueError('Destination appeared during installation; existing files were preserved.')
        prepared.rename(destination)
    return {'status':'installed','delivery':str(destination),'verified_files':count}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--archive',type=Path,help='Use the downloaded runtime ZIP without network access.')
    group.add_argument('--url',help='HTTPS mirror of the same pinned ZIP; checksum must match.')
    parser.add_argument('--destination',type=Path,default=DESTINATION)
    args = parser.parse_args()
    try:
        descriptor = json.loads(DESCRIPTOR.read_text('utf8'))
        result = install(descriptor,args.destination,args.archive,args.url)
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:
        print(json.dumps({'status':'action_required','message':str(exc)},ensure_ascii=True))
        return 2
    print(json.dumps(result,ensure_ascii=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
