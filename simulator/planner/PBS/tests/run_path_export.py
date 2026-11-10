#!/usr/bin/env python3
"""Exercise the actual PBS exporter using its already-built native objects."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir',type=Path,default=Path('/build/pbs-build'))
parser.add_argument('--source-dir',type=Path,default=Path('/build/pbs'))
args=parser.parse_args()
objects=[str(p) for p in sorted((args.build_dir/'CMakeFiles/pbs.dir').rglob('*.o'))
         if p.name!='driver.cpp.o']
if not objects:
    parser.error('Build PBS before exercising its exporter.')
with tempfile.TemporaryDirectory(prefix='lsmart-path-export-') as directory:
    binary=str(Path(directory)/'path_export')
    subprocess.run(['g++','-std=c++14','-D_GLIBCXX_ASSERTIONS','-pthread',
        '-I'+str(args.source_dir/'inc'),str(args.source_dir/'tests/path_export.cpp'),
        *objects,'-L/usr/local/lib','-lrpc','-lboost_program_options','-lboost_system',
        '-lboost_filesystem','-lspdlog','-o',binary],check=True)
    subprocess.run([binary],check=True,timeout=30)
