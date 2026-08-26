FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages needed for pixi, Java, and downloading assets.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    openjdk-21-jdk-headless \
    unzip \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Install pixi as the environment manager used by this repository.
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/opt/pixi sh \
    && ln -s /opt/pixi/bin/pixi /usr/local/bin/pixi

ENV PATH="/opt/pixi/bin:${PATH}"

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

# Install SemML when the submodule is present in the build context.
RUN if [ -f semml/build.py ]; then \
    pixi run -e gpu python semml/build.py \
    && ln -s "$(pwd)/semml" dependencies/semml; \
    else \
    echo "SemML submodule not present; skipping SemML installation."; \
    fi

# Default to an interactive shell; run experiments with:
# pixi run -e gpu python scripts/train.py alg=struct_ltl env=warehouse run=tmp
CMD ["bash"]
