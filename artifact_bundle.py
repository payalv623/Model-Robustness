"""Package heavy files for TeraBox and safely restore manually downloaded parts.

No credentials, cloud API calls, or deletion of source files are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'project_metadata/artifact_bundles.json'
INDEX = '__artifact_index__.json'
BLOCK = 1024 * 1024


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def safe(root, relative):
    root = root.resolve()
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or '\\' in relative or ':' in relative or not value.parts:
        raise ValueError(f'Unsafe artifact path: {relative}')
    path = root.joinpath(*value.parts)
    if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError(f'Symlink destination rejected: {relative}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact escapes destination: {relative}')
    return path


def collect(root, item):
    base = root / item['local_path']
    if not base.exists():
        raise FileNotFoundError(f'{base}: retrieve the Drive-only checkpoint first')
    candidates = [base] if base.is_file() else base.glob('*.zip') if item['id'] == 'transfer_archives' else base.rglob('*')
    result = []
    for path in sorted(candidates):
        if path.is_symlink() or not path.is_file() or '__pycache__' in path.parts or path.name == '.DS_Store':
            continue
        result.append(path)
    if not result:
        raise ValueError('No files selected')
    return result


def plan(files, size):
    if len(files) == 1 and files[0].stat().st_size > size:
        return [('raw', [(files[0], start, min(size, files[0].stat().st_size - start))])
                for start in range(0, files[0].stat().st_size, size)]
    batches, batch, total = [], [], 0
    for path in files:
        n = path.stat().st_size
        if n > size:
            raise ValueError(f'{path.name} exceeds part size; increase --part-mb or package it separately')
        if batch and total + n > size:
            batches.append(('zip', batch)); batch, total = [], 0
        batch.append((path, 0, n)); total += n
    if batch:
        batches.append(('zip', batch))
    return batches


def make_zip(target, root, batch):
    entries = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, _, n in batch:
            relative = source.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            digest, count = hashlib.sha256(), 0
            with source.open('rb') as reader, archive.open(info, 'w', force_zip64=True) as writer:
                while block := reader.read(BLOCK):
                    writer.write(block); digest.update(block); count += len(block)
            if count != n:
                raise ValueError(f'Source changed while packaging: {relative}')
            entries.append({'path': relative, 'bytes': count, 'sha256': digest.hexdigest()})
        archive.writestr(INDEX, json.dumps(entries, sort_keys=True))


def pack(root, item, destination, manifest_path, part_bytes, only_part=None):
    files = collect(root, item)
    signature = hashlib.sha256(json.dumps([(p.relative_to(root).as_posix(), p.stat().st_size,
        p.stat().st_mtime_ns) for p in files]).encode()).hexdigest()
    batches = plan(files, part_bytes)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'schema_version': 1, 'provider': 'terabox', 'folder_url': None, 'groups': {}}
    old = manifest['groups'].get(item['id'])
    if old and (old['source_snapshot'] != signature or old['part_bytes'] != part_bytes):
        raise ValueError('Source snapshot or part size changed; use a new manifest/export revision')
    group = old or {'source_snapshot': signature, 'part_bytes': part_bytes,
        'expected_parts': len(batches), 'files': len(files), 'parts': [], 'prepared_complete': False}
    selected = [only_part] if only_part is not None else list(range(1, len(batches) + 1))
    if any(i < 1 or i > len(batches) for i in selected):
        raise ValueError(f'Part number must be between 1 and {len(batches)}')
    destination.mkdir(parents=True, exist_ok=True)
    for number in selected:
        kind, batch = batches[number - 1]
        name = f"{item['id']}-{number:04d}-of-{len(batches):04d}." + ('zip' if kind == 'zip' else 'part')
        target = destination / name
        previous = next((p for p in group['parts'] if p['number'] == number), None)
        if previous and target.exists() and sha(target) == previous['sha256']:
            print('Verified existing', name, flush=True); continue
        needed = sum(n for _, _, n in batch)
        if shutil.disk_usage(destination).free < needed + 600_000_000:
            raise OSError('Insufficient staging space; upload prepared parts before creating more')
        temporary = target.with_suffix(target.suffix + '.partial')
        if kind == 'zip':
            make_zip(temporary, root, batch)
        else:
            source, start, length = batch[0]
            with source.open('rb') as reader, temporary.open('wb') as writer:
                reader.seek(start)
                remaining = length
                while remaining:
                    block = reader.read(min(BLOCK, remaining))
                    if not block:
                        raise ValueError('Source truncated during packaging')
                    writer.write(block); remaining -= len(block)
        checksum = sha(temporary)
        if previous and checksum != previous['sha256']:
            raise ValueError('Regenerated archive differs from the recorded part')
        temporary.replace(target)
        part = {'number': number, 'file': name, 'kind': kind, 'bytes': target.stat().st_size,
            'sha256': checksum, 'upload_status': 'pending', 'remote_path': None}
        if kind == 'raw':
            part.update({'restore_path': batch[0][0].relative_to(root).as_posix(), 'offset': batch[0][1],
                         'total_bytes': batch[0][0].stat().st_size})
        group['parts'] = sorted([p for p in group['parts'] if p['number'] != number] + [previous or part], key=lambda p:p['number'])
        group['prepared_complete'] = len(group['parts']) == len(batches)
        manifest['groups'][item['id']] = group
        save(manifest_path, manifest)
        print('Prepared', name, target.stat().st_size, 'bytes', flush=True)
    return group


def checked_zip(path, destination):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count(INDEX) != 1:
            raise ValueError('Archive index missing or duplicated')
        entries = json.loads(archive.read(INDEX))
        expected = [e['path'] for e in entries]
        if len(expected) != len(set(expected)) or sorted(archive.namelist()) != sorted(expected + [INDEX]):
            raise ValueError('Unexpected or duplicate archive members')
        for entry in entries:
            target = safe(destination, entry['path'])
            info = archive.getinfo(entry['path'])
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000 or info.file_size != entry['bytes']:
                raise ValueError('Invalid archive member')
            if target.exists() and (not target.is_file() or sha(target) != entry['sha256']):
                raise ValueError(f'Refusing to replace changed local file: {entry["path"]}')
        return entries


def restore(manifest, download_dir, destination, groups=None):
    selected = groups or list(manifest['groups'])
    prepared, destinations = [], {}
    for name in selected:
        group = manifest['groups'][name]
        parts = group['parts']
        if not group['prepared_complete'] or [p['number'] for p in parts] != list(range(1, group['expected_parts'] + 1)):
            raise ValueError(f'Incomplete group: {name}')
        for part in parts:
            path = safe(download_dir, part['file'])
            if not path.is_file() or path.stat().st_size != part['bytes'] or sha(path) != part['sha256']:
                raise ValueError(f'Missing or corrupted downloaded part: {part["file"]}')
            entries = checked_zip(path, destination) if part['kind'] == 'zip' else None
            if part['kind'] not in ('zip', 'raw'):
                raise ValueError('Unknown bundle kind')
            if entries is not None:
                for entry in entries:
                    previous = destinations.setdefault(entry['path'], entry['sha256'])
                    if previous != entry['sha256']:
                        raise ValueError('Conflicting archive members across parts')
            else:
                safe(destination, part['restore_path'])
            prepared.append((name, part, path, entries))
    destination.mkdir(parents=True, exist_ok=True)
    # Every archive is hash-checked and every ZIP path is preflighted before writing.
    for _, part, path, entries in prepared:
        if entries is None:
            continue
        with zipfile.ZipFile(path) as archive:
            for entry in entries:
                target = safe(destination, entry['path'])
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as writer:
                    temporary = Path(writer.name)
                    try:
                        with archive.open(entry['path']) as reader:
                            shutil.copyfileobj(reader, writer, BLOCK)
                    except BaseException:
                        temporary.unlink(missing_ok=True); raise
                if sha(temporary) != entry['sha256']:
                    temporary.unlink(); raise ValueError('Restored member checksum mismatch')
                temporary.replace(target)
        print('Restored', part['file'], flush=True)
    for name in selected:
        parts = [x for x in prepared if x[0] == name and x[1]['kind'] == 'raw']
        if not parts:
            continue
        target = safe(destination, parts[0][1]['restore_path'])
        offset = 0
        for _, part, _, _ in parts:
            if part['offset'] != offset or part['restore_path'] != parts[0][1]['restore_path']:
                raise ValueError('Non-contiguous raw parts')
            offset += part['bytes']
        if offset != parts[0][1]['total_bytes']:
            raise ValueError('Incomplete raw file')
        if target.exists():
            # Comparing each source slice avoids overwriting an unrelated local file.
            with target.open('rb') as reader:
                for _, part, _, _ in parts:
                    digest, remaining = hashlib.sha256(), part['bytes']
                    while remaining:
                        block = reader.read(min(BLOCK, remaining))
                        if not block: raise ValueError('Existing raw file is truncated')
                        digest.update(block); remaining -= len(block)
                    if digest.hexdigest() != part['sha256']: raise ValueError('Existing raw file differs')
                if reader.read(1): raise ValueError('Existing raw file has extra data')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as writer:
            temporary = Path(writer.name)
            try:
                for _, _, path, _ in parts:
                    with path.open('rb') as reader: shutil.copyfileobj(reader, writer, BLOCK)
            except BaseException:
                temporary.unlink(missing_ok=True); raise
        temporary.replace(target)
        print('Reassembled', target.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    sub = parser.add_subparsers(dest='command', required=True)
    packer = sub.add_parser('pack')
    packer.add_argument('group')
    packer.add_argument('--output', type=Path, default=ROOT / 'artifact_exports')
    packer.add_argument('--part-mb', type=int, default=1000)
    packer.add_argument('--part', type=int, help='Prepare only this one-based part; useful with limited disk space')
    restorer = sub.add_parser('restore')
    restorer.add_argument('--downloads', type=Path, required=True)
    restorer.add_argument('--destination', type=Path, default=ROOT)
    restorer.add_argument('--group', action='append')
    args = parser.parse_args()
    if args.command == 'pack':
        if args.part_mb < 1: parser.error('--part-mb must be positive')
        inventory = json.loads((ROOT / 'project_metadata/artifact_inventory.json').read_text())
        item = next((i for i in inventory['artifacts'] if i['id'] == args.group), None)
        if item is None: parser.error('Unknown group')
        pack(ROOT, item, args.output, args.manifest, args.part_mb * 1_000_000, args.part)
    else:
        restore(json.loads(args.manifest.read_text()), args.downloads.resolve(), args.destination.resolve(), args.group)


if __name__ == '__main__':
    main()
