#!/usr/bin/env python3
"""Link parking regressions against the server and optionally replay a request."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, default=Path('/build/server-build'))
parser.add_argument('--source-dir', type=Path, default=Path('/build/server'))
parser.add_argument('--backup-source-dir', type=Path,
                    help='Directory containing alternative PIBT.cpp/GuidedPIBT.cpp, for red/green replay.')
parser.add_argument('--replay-map', type=Path)
parser.add_argument('--replay-request', type=Path)
parser.add_argument('--replay-tasks', type=Path)
parser.add_argument('--replay-output', type=Path)
args = parser.parse_args()
replay = [args.replay_map, args.replay_request, args.replay_tasks, args.replay_output]
if any(replay) and not all(replay):
    parser.error('Replay requires --replay-map, --replay-request, --replay-tasks, and --replay-output.')
objects = sorted((args.build_dir / 'CMakeFiles/ExecutionManager.dir').rglob('*.o'))
objects = [str(path) for path in objects
           if path.name not in {'server.cpp.o', 'PIBT.cpp.o', 'GuidedPIBT.cpp.o'}]
if not objects:
    parser.error('Build the native server before testing.')
with tempfile.TemporaryDirectory(prefix='lsmart-parking-test-') as directory:
    binary = str(Path(directory) / 'pibt_parking')
    backup = args.backup_source_dir or args.source_dir / 'src/backup_planners'
    subprocess.run([
        'g++', '-std=c++2a', '-D_GLIBCXX_ASSERTIONS', '-pthread',
        '-I' + str(args.source_dir / 'inc'),
        str(args.source_dir / 'tests/pibt_parking.cpp'),
        str(backup / 'PIBT.cpp'), str(backup / 'GuidedPIBT.cpp'), *objects,
        '-L/usr/local/lib', '-lrpc', '-lboost_program_options',
        '-lboost_system', '-lboost_filesystem', '-lspdlog', '-o', binary,
    ], check=True)
    subprocess.run([binary, *[str(path) for path in replay if path]], check=True, timeout=180)
