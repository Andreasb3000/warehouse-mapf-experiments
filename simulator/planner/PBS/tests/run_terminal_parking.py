#!/usr/bin/env python3
"""Test native PBS terminal parking; optionally substitute pre-fix sources."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, default=Path('/build/pbs-build'))
parser.add_argument('--source-dir', type=Path, default=Path('/build/pbs'))
parser.add_argument('--legacy-source-dir', type=Path,
                    help='Directory containing pre-fix Instance/SingleAgentSolver/SpaceTimeAStar/SIPP.cpp.')
parser.add_argument('--solver', choices=['both', 'ASTAR', 'SIPP'], default='both')
args = parser.parse_args()
names = ['Instance.cpp', 'SingleAgentSolver.cpp', 'SpaceTimeAStar.cpp', 'SIPP.cpp']
objects = [str(path) for path in sorted((args.build_dir / 'CMakeFiles/pbs.dir').rglob('*.o'))
           if path.name not in {'driver.cpp.o', *[name + '.o' for name in names]}]
if not objects:
    parser.error('Build PBS before running its parking regressions.')
sources = args.legacy_source_dir or args.source_dir / 'src'
with tempfile.TemporaryDirectory(prefix='lsmart-pbs-parking-test-') as directory:
    binary = str(Path(directory) / 'terminal_parking')
    subprocess.run([
        'g++', '-std=c++14', '-D_GLIBCXX_ASSERTIONS', '-pthread',
        '-I' + str(args.source_dir / 'inc'),
        str(args.source_dir / 'tests/terminal_parking.cpp'),
        *[str(sources / name) for name in names], *objects,
        '-L/usr/local/lib', '-lrpc', '-lboost_program_options',
        '-lboost_system', '-lboost_filesystem', '-lspdlog', '-o', binary,
    ], check=True)
    subprocess.run([binary, args.solver], check=True, timeout=60)
