# Third-party notices

## LSMART

Source: https://github.com/smart-mapf/lifelong-smart

Upstream checkout revision: `affaa43db4faf2595e2126fe26234ce475eee596`.

The exported simulator includes local changes for finite tasks, safety geometry, predictor snapshots, planner penalties, candidate-plan preview/installation, admission controls and terminal parking. The upstream MIT text is retained in `simulator/LICENSE.txt`.

## PBS

The planner source in `simulator/planner/PBS` retains its University of Southern California copyright and educational/research/non-profit license in `license.md`. Those terms govern that component, including its retained source and modifications.

## rpclib and vendored dependencies

The RPC library in `simulator/client/externalDependencies/rpclib` retains its license and bundled dependency notices. Source headers and license files are preserved.

## Runtime dependencies

Docker builds fetch the public LSMART runtime image by digest and the official ARGoS beta59 package by verified SHA-256. These binary dependencies are not included in the source export. Python dependencies are installed separately under their own licenses.

