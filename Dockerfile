# syntax=docker/dockerfile:1
# CPU-only development image for ROS 2 Jazzy. No GPU, no datasets, no weights.
# The base tag was verified against Docker Hub on 2026-10-30 (HTTP 200).
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.13.6

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash sim2real
USER sim2real
WORKDIR /home/sim2real/ws

# Pin uv to the version verified in this project's preflight.
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
ENV PATH="/home/sim2real/.local/bin:${PATH}"

CMD ["/bin/bash"]
