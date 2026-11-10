#!/usr/bin/env python3
"""Preserve frozen gate sources and provenance without changing a running study.

The source snapshot can be created while simulations run: only immutable
manifest fields are compared. This is the manifest's hashed Python/native
source subset plus reporting tools, not an export of the Docker image or a
complete standalone checkout. Existing snapshots are verified, never replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SERVER_PREFIX = Path('simulator/server')
INDEX_NAME = 'archive_manifest.json'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path == Path('.'):
        raise ValueError(f'Expected a workspace-relative file path: {value!r}')
    return path


def protocol(manifest):
    """Run status/error may evolve; policy, image, sources and run list may not."""
    return {key: value for key, value in manifest.items() if key not in ('status', 'error')}


def frozen_files(manifest):
    if not manifest.get('source_sha256'):
        raise ValueError('Missing frozen Python implementation hashes')
    files = {str(relative_path(name)): value for name, value in manifest['source_sha256'].items()}
    for name, value in manifest.get('compiled_server_sha256', {}).items():
        target = str(SERVER_PREFIX / relative_path(name))
        if target in files and files[target] != value:
            raise ValueError(f'Conflicting frozen hashes: {target}')
        files[target] = value
    return files


def verification_matches(verification, manifest):
    """Bind fresh unified build proof (or a legacy archived proof) to a study."""
    if 'compiled_source_sha256' in verification:
        compiled = verification['compiled_source_sha256']
        server = {key.removeprefix('server/'): value for key, value in compiled.items()
                  if key.startswith('server/')}
        return (verification.get('image_id') == manifest['image_id'] and bool(server)
                and all(server.get(key) == value for key, value
                        in manifest.get('compiled_server_sha256', {}).items())
                and bool(manifest.get('compiled_server_sha256')))
    return (verification.get('status') == 'verified'
            and verification.get('gate_image_id') == manifest['image_id']
            and verification.get('gate_sources') == manifest.get('compiled_server_sha256', {}))


def verify_source_snapshot(study, manifest=None):
    """Check archived source bytes and bind provenance to the live manifest."""
    study = Path(study)
    manifest = read(study / 'manifest.json') if manifest is None else manifest
    snapshot = study / 'source_snapshot'
    archive = read(snapshot / INDEX_NAME)
    if archive.get('schema_version') != 1:
        raise ValueError('Unsupported source snapshot format')
    if archive.get('image_id') != manifest['image_id']:
        raise ValueError('Archived native image differs from study manifest')
    if read(snapshot / 'metadata/protocol.json') != protocol(manifest):
        raise ValueError('Archived protocol differs from immutable study fields')
    files = archive.get('files_sha256', {})
    for name, expected in frozen_files(manifest).items():
        if files.get(name) != expected:
            raise ValueError(f'Archive omits or changes a frozen source hash: {name}')
    for name in ('metadata/protocol.json', 'metadata/study_manifest_at_archive.json',
                 'metadata/native_build_proof.json'):
        if name not in files:
            raise ValueError(f'Archive omits required provenance: {name}')
    for name, expected in files.items():
        path = snapshot / relative_path(name)
        if digest(path) != expected:
            raise ValueError(f'Archived file changed: {name}')
    verification = read(snapshot / 'metadata/native_build_proof.json')
    if not verification_matches(verification, manifest):
        raise ValueError('Archived native verification does not match image/source hashes')
    if protocol(read(snapshot / 'metadata/study_manifest_at_archive.json')) != protocol(manifest):
        raise ValueError('Archived original manifest does not match frozen study protocol')
    return archive


def archive_study(study, *, workspace=ROOT.parent, native_verification=None,
                  smoke_audit=None, include_report_tools=True):
    study, workspace = Path(study), Path(workspace)
    manifest_bytes = (study / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    destination = study / 'source_snapshot'
    if destination.exists():
        verify_source_snapshot(study, manifest)
        return destination
    native_verification = Path(native_verification or ROOT / 'native/build_proof.json')
    smoke_audit = Path(smoke_audit) if smoke_audit is not None else None
    verification = read(native_verification)
    if not verification_matches(verification, manifest):
        raise ValueError('Native verification must match the frozen image and compiled source hashes')
    files = frozen_files(manifest)
    # Validate every current execution source before creating any archive tree.
    for name, expected in files.items():
        if digest(workspace / name) != expected:
            raise ValueError(f'Workspace differs from frozen execution source: {name}')
    helpers = []
    if include_report_tools:
        helpers = [Path(__file__), Path(__file__).with_name('build_entry_gate_results.py')]
        for helper in helpers:
            name = str(helper.resolve().relative_to(workspace.resolve()))
            files[name] = digest(helper)
    with tempfile.TemporaryDirectory(prefix='.source_snapshot-', dir=study) as temporary:
        temporary = Path(temporary)
        staged = temporary / 'source_snapshot'
        staged.mkdir()
        for name, expected in files.items():
            target = staged / relative_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace / name, target)
            if digest(target) != expected:
                raise ValueError(f'Source changed while archiving: {name}')
        metadata = staged / 'metadata'; metadata.mkdir()
        (metadata / 'study_manifest_at_archive.json').write_bytes(manifest_bytes)
        (metadata / 'protocol.json').write_text(json.dumps(protocol(manifest), indent=2) + '\n')
        shutil.copy2(native_verification, metadata / 'native_build_proof.json')
        if smoke_audit is not None:
            shutil.copy2(smoke_audit, metadata / 'mechanism_smoke_audit.json')
        for path in metadata.iterdir():
            files[str(path.relative_to(staged))] = digest(path)
        archive = {
            'schema_version': 1,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'relative_to': 'workspace root for sources; metadata/ for provenance',
            'image_id': manifest['image_id'],
            'source_file_count': len(manifest['source_sha256']),
            'compiled_server_source_file_count': len(manifest.get('compiled_server_sha256', {})),
            'report_tool_paths': [str(path.resolve().relative_to(workspace.resolve())) for path in helpers],
            'files_sha256': dict(sorted(files.items())),
            'scope': 'Frozen execution source subset, reporting helpers, immutable protocol, native build verification and an optional supplied mechanism smoke audit. Docker image bytes and unlisted dependencies are not exported.',
        }
        (staged / INDEX_NAME).write_text(json.dumps(archive, indent=2) + '\n')
        if protocol(read(study / 'manifest.json')) != protocol(manifest):
            raise ValueError('Frozen study protocol changed during archiving')
        staged.rename(destination)
    verify_source_snapshot(study)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'data/entry_gates/central_gate_v3')
    parser.add_argument('--native-verification', type=Path, default=ROOT / 'native/build_proof.json')
    parser.add_argument('--smoke-audit', type=Path, help='Optional independently produced mechanism audit')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        archive = verify_source_snapshot(args.study)
        print(json.dumps({'snapshot': str(args.study / 'source_snapshot'), 'verified_files': len(archive['files_sha256'])}))
    else:
        print(archive_study(args.study, native_verification=args.native_verification, smoke_audit=args.smoke_audit))


if __name__ == '__main__':
    main()
