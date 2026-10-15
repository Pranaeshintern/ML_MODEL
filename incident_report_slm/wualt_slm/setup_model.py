"""Download pinned public assets into this project and verify their SHA256."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

from .settings import ASSET_ROOT


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == digest:
        print(f'Already verified: {target.name}', flush=True)
        return
    partial = target.with_suffix(target.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        append = response.status == 206 and offset > 0
        if not append:
            offset = 0
        last_print = time.monotonic()
        with partial.open('ab' if append else 'wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                offset += len(chunk)
                if time.monotonic() - last_print > 10:
                    print(f'{target.name}: {offset / 1e6:.0f} MB downloaded', flush=True)
                    last_print = time.monotonic()
    if sha256(partial) != digest:
        raise RuntimeError(f'Checksum mismatch for {target.name}; remove the .partial file and retry.')
    partial.replace(target)
    print(f'Verified: {target.name}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path.cwd())
    parser.add_argument('--model-only', action='store_true', help='Download weights without the Windows runtime')
    args = parser.parse_args(argv)
    if os.name != 'nt' and not args.model_only:
        parser.error('Use --model-only on Linux/macOS, then supply your own llama.cpp server.')
    root = args.directory.resolve()
    lock = json.loads((ASSET_ROOT / 'runtime.lock.json').read_text(encoding='utf-8'))
    if not args.model_only:
        runtime = root / '.runtime'
        archive = runtime / 'llama-vulkan.zip'
        download(lock['runtime_url'], archive, lock['runtime_sha256'])
        destination = (runtime / 'llama').resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise ValueError('Unsafe archive path')
            z.extractall(destination)
    url = f"https://huggingface.co/{lock['model_id']}/resolve/{lock['model_revision']}/{lock['model_file']}"
    download(url, root / 'models' / lock['model_file'], lock['model_sha256'])
    print('Pinned model assets are verified.', flush=True)


if __name__ == '__main__':
    main()
