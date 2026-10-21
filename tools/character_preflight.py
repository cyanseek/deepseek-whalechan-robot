#!/usr/bin/env python3
"""Offline JSON readiness check for the VD24 source demo; does not run physics."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def check(delivery):
    report = {'schema_version': 1, 'profile': 'VD24',
              'python': platform.python_version(), 'platform': platform.system(),
              'dependencies': {}, 'errors': [], 'physics_executed': False,
              'rendering_checked': False, 'asset_permission_verified': False}
    if tuple(__import__('sys').version_info[:2]) < (3, 11):
        report['errors'].append('Python 3.11 or newer is required; verified environment uses 3.13.5.')
    packages = ['numpy', 'mujoco', 'opencv-python', 'Pillow', 'imageio', 'imageio-ffmpeg', 'scipy']
    for package in packages:
        try:
            report['dependencies'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['errors'].append('Missing dependency: ' + package)
    if report['dependencies'].get('mujoco') not in (None, '3.3.7'):
        report['errors'].append('This profile requires MuJoCo 3.3.7.')
    delivery = Path(delivery).resolve()
    required = ['BlueFatFish_Visual.glb', 'DELIVERY_MANIFEST.json',
                'mujoco_preview/manifest.json', 'mujoco_preview/visual_mocap.xml',
                'credits/THIRD_PARTY_NOTICES.md']
    missing = [name for name in required if not (delivery / name).is_file()]
    report['missing_files'] = missing
    if missing:
        report['errors'].append('Incomplete BlueFatFish visual delivery; see docs/ASSETS.md.')
    else:
        try:
            manifest = json.loads((delivery / 'mujoco_preview/manifest.json').read_text('utf8'))
            receipt = json.loads((delivery / 'DELIVERY_MANIFEST.json').read_text('utf8'))
            entries = receipt.get('files', receipt) if isinstance(receipt, dict) else receipt
            if isinstance(entries, dict):
                entries = [dict(path=k, **v) for k, v in entries.items()]
            if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
                raise ValueError('Delivery manifest must contain file records')
            verified = 0
            for item in entries:
                name = item.get('path', item.get('file', ''))
                if not isinstance(name, str):
                    raise ValueError('Manifest path must be a string')
                if not name.startswith('mujoco_preview/'):
                    continue
                path = (delivery / name).resolve()
                if not path.is_relative_to(delivery):
                    raise ValueError('Manifest path escapes the selected delivery')
                if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                    raise ValueError('Asset hash mismatch: ' + name)
                verified += 1
            if not verified:
                raise ValueError('Delivery manifest contains no preview asset hashes')
            if hashlib.sha256((delivery / 'BlueFatFish_Visual.glb').read_bytes()).hexdigest() != manifest['source_glb_sha256']:
                raise ValueError('GLB hash differs from the skin manifest')
            report['verified_preview_files'] = verified
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report['errors'].append('Asset validation failed: ' + str(exc))
    report['status'] = 'ready_for_native_check' if not report['errors'] else 'action_required'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True,
                        help='Directory containing BlueFatFish_Visual.glb and mujoco_preview/')
    args = parser.parse_args()
    report = check(args.delivery)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report['status'] == 'ready_for_native_check' else 2


if __name__ == '__main__':
    raise SystemExit(main())
