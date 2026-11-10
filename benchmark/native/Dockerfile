# Rebuild every modified native component from this checkout. The public base
# supplies ARGoS beta59, its visualizer and runtime dependencies.
ARG BASE_IMAGE=lunjohnzhang/lsmart@sha256:4dad94f9f9be52d22820be540dfaf0c74f07f5c8232e684fc03f8bc78f974657
FROM ${BASE_IMAGE} AS native-build
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config freeglut3-dev libboost-program-options-dev \
    libboost-system-dev libboost-filesystem-dev libfreeimage-dev libfreeimageplus-dev \
    liblua5.3-dev libspdlog-dev libxi-dev libxmu-dev qtbase5-dev libqt5opengl5-dev \
    && rm -rf /var/lib/apt/lists/*
COPY .local/argos3-beta59-ubuntu20.deb /tmp/argos3.deb
RUN echo '2bcd19430266e80a54a053d9b5079212468f5f518042921eadb7cbd2efa71fd8  /tmp/argos3.deb' | sha256sum -c - \
    && dpkg-deb -x /tmp/argos3.deb /tmp/argos-dev \
    && cp -a /tmp/argos-dev/usr/include/argos3 /usr/include/ \
    && cp -a /tmp/argos-dev/usr/share/argos3 /usr/share/
COPY client /build/client
RUN cmake -S /build/client -B /build/client-build -DCMAKE_BUILD_TYPE=Release \
        -DARGoS_DIR=/usr/share/argos3/cmake \
        '-DCMAKE_BUILD_RPATH=/usr/project/client/build/utils;/usr/project/client/build/controllers/footbot_diffusion' \
    && cmake --build /build/client-build --parallel 2 \
    && cmake --install /build/client-build/externalDependencies/rpclib \
    && c++ -std=c++17 -I/build/client /build/client/tests/test_safety_geometry.cpp -o /build/test-safety \
    && /build/test-safety
COPY server /build/server
RUN cmake -S /build/server -B /build/server-build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build/server-build --parallel 2 \
    && python3 /build/server/tests/run_finite_tasks.py \
    && python3 /build/server/tests/run_entry_gate.py \
    && python3 /build/server/tests/run_admission_bridge.py \
    && python3 /build/server/tests/run_route_preview.py \
    && python3 /build/server/tests/run_pibt_parking.py
COPY planner/PBS /build/planner/PBS
RUN cmake -S /build/planner/PBS -B /build/pbs-build -DCMAKE_BUILD_TYPE=Release -DLSMART_BUILD_ROUTING_TESTS=ON \
    && cmake --build /build/pbs-build --parallel 2 \
    && /build/pbs-build/routing_penalties_test \
    && /build/pbs-build/route_search_test \
    && python3 /build/planner/PBS/tests/run_path_export.py --source-dir /build/planner/PBS \
    && python3 /build/planner/PBS/tests/run_terminal_parking.py --source-dir /build/planner/PBS
COPY native_manifest.py /build/native_manifest.py
RUN python3 /build/native_manifest.py /build /build/manifests

FROM ${BASE_IMAGE}
COPY --from=native-build --chown=lsmart:lsmart /build/server-build/ExecutionManager /usr/project/server/build/ExecutionManager
COPY --from=native-build --chown=lsmart:lsmart /build/client-build/controllers/footbot_diffusion/libfootbot_diffusion.so /usr/project/client/build/controllers/footbot_diffusion/libfootbot_diffusion.so
COPY --from=native-build --chown=lsmart:lsmart /build/client-build/loop_functions/trajectory_loop_functions/libtrajectory_loop_functions.so /usr/project/client/build/loop_functions/trajectory_loop_functions/libtrajectory_loop_functions.so
COPY --from=native-build --chown=lsmart:lsmart /build/client-build/utils/libutils.so /usr/project/client/build/utils/libutils.so
COPY --from=native-build --chown=lsmart:lsmart /build/pbs-build/pbs /usr/project/planner/PBS/build/pbs
COPY --from=native-build --chown=lsmart:lsmart /build/manifests/ /usr/project/
COPY --chown=lsmart:lsmart run_lifelong.py run_lifelong_kwargs.py /usr/project/
COPY --chown=lsmart:lsmart ArgosConfig /usr/project/ArgosConfig
COPY --chown=lsmart:lsmart experiments/junction/scenario.py /usr/project/experiments/junction/scenario.py
LABEL org.opencontainers.image.title="Warehouse congestion experiments" \
      org.opencontainers.image.description="Predictor and planning baselines with current terminal-parking correction"
