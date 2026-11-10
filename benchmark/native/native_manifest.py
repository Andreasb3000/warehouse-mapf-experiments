"""Write image source manifests from the actual native build inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifests(root):
    def source_file(path):
        parts = path.relative_to(root).parts
        return ('build' not in parts and '__pycache__' not in parts
                and not any(part.startswith('cmake-build-') for part in parts))

    def hashed(paths):
        return {str(p.relative_to(root)): digest(p) for p in sorted(set(paths)) if source_file(p)}

    server = [*(root / 'server/src').rglob('*.cpp'), *(root / 'server/inc').rglob('*.h')]
    client = [p for p in (root / 'client').rglob('*')
              if p.suffix in ('.cpp', '.h') and 'externalDependencies' not in p.parts]
    pbs = [*(root / 'planner/PBS/src').rglob('*.cpp'), *(root / 'planner/PBS/inc').rglob('*.h')]
    route = [p for folder in ('server', 'planner/PBS') for p in (root / folder).rglob('*')
             if p.is_file() and (p.suffix in ('.cpp', '.h', '.py') or p.name == 'CMakeLists.txt')]
    route += [p for p in (root / 'client').rglob('*') if p.is_file()
              and 'externalDependencies' not in p.parts
              and (p.suffix in ('.cpp', '.h', '.py') or p.name == 'CMakeLists.txt')]
    gate = {str(p.relative_to(root / 'server')): digest(p)
            for p in sorted((root / 'server').rglob('*'))
            if source_file(p) and p.suffix in ('.cpp', '.h', '.py')}
    return {
        'server-source-sha256.json': hashed(server + client),
        'pbs-source-sha256.json': hashed(pbs),
        'routing-source-sha256.json': hashed(server + client + pbs),
        'route-search-source-sha256.json': hashed(route),
        'entry-gate-source-sha256.json': gate,
        'admission-source-sha256.json': gate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in manifests(args.source).items():
        (args.output / name).write_text(json.dumps(content, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
