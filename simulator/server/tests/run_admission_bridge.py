#!/usr/bin/env python3
"""Exercise accepted-plan admission on the native primary and fallback paths."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, default=Path('/build/server-build'))
parser.add_argument('--source-dir', type=Path, default=Path('/build/server'))
args = parser.parse_args()
objects = sorted((args.build_dir/'CMakeFiles/ExecutionManager.dir').rglob('*.o'))
objects = [str(path) for path in objects if path.name != 'server.cpp.o']
if not objects:
    parser.error('Build the native server before testing.')
with tempfile.TemporaryDirectory(prefix='lsmart-admission-test-') as directory:
    binary = str(Path(directory)/'admission_bridge')
    subprocess.run(['g++', '-std=c++2a', '-D_GLIBCXX_ASSERTIONS', '-pthread',
        '-I'+str(args.source_dir/'inc'), str(args.source_dir/'tests/admission_bridge.cpp'),
        *objects, '-L/usr/local/lib', '-lrpc', '-lboost_program_options',
        '-lboost_system', '-lboost_filesystem', '-lspdlog', '-o', binary], check=True)
    subprocess.run([binary], check=True, timeout=30)
