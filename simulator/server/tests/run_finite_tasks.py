#!/usr/bin/env python3
"""Link finite_tasks.cpp against an existing execution-server build and run it.

Inside the junction Dockerfile's server-build stage:
  python3 server/tests/run_finite_tasks.py

Tests are compiled with assertions enabled independently of the server's build.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("/build/server-build"))
    parser.add_argument("--source-dir", type=Path, default=Path("/build/server"))
    args = parser.parse_args()
    objects = sorted((args.build_dir / "CMakeFiles/ExecutionManager.dir").rglob("*.o"))
    # Recompile the conversion code with checked STL indexing. Its regression
    # must catch an out-of-bounds access even when Release happens to read
    # harmless-looking memory beyond a goal vector.
    objects = [str(path) for path in objects
               if path.name not in {"server.cpp.o", "FailPolicy.cpp.o"}]
    if not objects:
        parser.error("Build ExecutionManager before running these tests.")
    with tempfile.TemporaryDirectory(prefix="lsmart-finite-tests-") as directory:
        binary = str(Path(directory) / "finite_tasks")
        subprocess.run([
            "g++", "-std=c++2a", "-D_GLIBCXX_ASSERTIONS", "-pthread",
            "-I" + str(args.source_dir / "inc"),
            str(Path(__file__).with_name("finite_tasks.cpp")),
            str(args.source_dir / "src/backup_planners/FailPolicy.cpp"), *objects,
            "-L/usr/local/lib", "-lrpc", "-lboost_program_options",
            "-lboost_system", "-lboost_filesystem", "-lspdlog", "-o", binary,
        ], check=True)
        subprocess.run([binary], check=True)


if __name__ == "__main__":
    main()
