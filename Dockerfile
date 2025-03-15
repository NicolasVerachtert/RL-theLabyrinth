# Use the official lightweight Python DevContainer image
FROM mcr.microsoft.com/devcontainers/python:3.11

# Set environment variables
ENV MUJOCO_GL=egl
# ENV DISPLAY=:0

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first (to leverage Docker caching)
COPY requirements.txt .

# Install minimal system dependencies and clean up afterward
RUN --mount=type=cache,target=/var/cache/apt \
    apt update && \
    apt install -y --no-install-recommends libsdl2-dev libosmesa6 libopengl0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies without caching to reduce image size
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set root as the default user
USER root

# Define the default command
CMD ["/bin/bash"]
