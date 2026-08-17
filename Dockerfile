FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages needed for pixi, Java (Rabinizer), and downloading assets.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    openjdk-17-jre-headless \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install pixi as the environment manager used by this repository.
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/opt/pixi sh \
    && ln -s /opt/pixi/bin/pixi /usr/local/bin/pixi

WORKDIR /workspace

# Copy project files and install the GPU environment declared in pyproject.toml.
COPY . .
RUN pixi install -e gpu \
    && pixi run -e gpu copy-templates \
    && pixi clean cache -y

# Install Rabinizer 4 in the expected project location.
RUN mkdir -p dependencies \
    && curl -L https://www7.in.tum.de/~kretinsk/rabinizer4.zip -o /tmp/rabinizer4.zip \
    && unzip -q /tmp/rabinizer4.zip -d /tmp \
    && rm -rf dependencies/rabinizer4 \
    && mv /tmp/rabinizer4 dependencies/rabinizer4 \
    && rm -f /tmp/rabinizer4.zip

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="/opt/pixi/bin:${PATH}"

# Default to an interactive shell; run experiments with:
# pixi run -e gpu python scripts/train.py experiment=struct_ltl/warehouse run=tmp
CMD ["bash"]