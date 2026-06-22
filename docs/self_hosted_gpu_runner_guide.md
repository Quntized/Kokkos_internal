# Self-Hosted GPU Runner Setup Guide

Complete guide for running your Kokkos CUDA tests on a self-hosted GitHub Actions runner.

## Prerequisites

On your GPU machine, verify you have:

```bash
# 1. NVIDIA driver installed and working
nvidia-smi

# 2. CUDA toolkit (should already be there if you've been building Kokkos locally)
nvcc --version

# 3. Docker installed (recommended approach)
docker --version

# 4. NVIDIA Container Toolkit (for passing GPUs into Docker containers)
# Install if missing:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify Docker can see the GPU:
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

---

## Step 1: Register the Runner on GitHub

### 1a. Go to your repository settings

Navigate to:
```
https://github.com/<your-org-or-user>/kokkos_internal/settings/actions/runners/new
```

Or: **Repo → Settings → Actions → Runners → New self-hosted runner**

### 1b. Download and configure the runner

GitHub will show you platform-specific commands. For Linux x64, it'll look like:

```bash
# Create a directory for the runner
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download the latest runner package (GitHub will show the exact URL)
curl -o actions-runner-linux-x64-2.321.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-x64-2.321.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.321.0.tar.gz
```

### 1c. Configure with labels

```bash
# The token comes from the GitHub settings page above
./config.sh \
  --url https://github.com/<your-org-or-user>/kokkos_internal \
  --token <TOKEN_FROM_GITHUB_UI> \
  --name "gpu-runner" \
  --labels self-hosted,linux,x64,gpu \
  --work _work
```

> [!IMPORTANT]
> The `--labels gpu` part is key — your workflow will target this label to route CUDA jobs to this runner specifically.

---

## Step 2: Start the Runner

### Option A: Interactive (for testing)

```bash
./run.sh
```

You'll see output like:
```
√ Connected to GitHub
Listening for Jobs
```

### Option B: As a systemd service (for production)

```bash
# Install the service (run from the actions-runner directory)
sudo ./svc.sh install

# Start it
sudo ./svc.sh start

# Check status
sudo ./svc.sh status

# View logs
journalctl -u actions.runner.<repo-name>.gpu-runner.service -f
```

> [!TIP]
> The systemd service will auto-start on boot and restart on failure — this is the recommended approach.

---

## Step 3: Update Your CI Workflow

Update your `.github/workflows/ci.yml` to add a GPU test job:

```yaml
  cuda-test:
    name: "Build & Test (CUDA 12.6, GPU)"
    runs-on: [self-hosted, gpu, linux]
    container:
      image: nvidia/cuda:12.6.3-devel-ubuntu24.04
      options: --gpus all    # <-- critical: passes host GPU into container

    env:
      BUILD_DIR: build
      CMAKE_BUILD_PARALLEL_LEVEL: 4

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install build tools
        run: |
          apt-get update -qq
          apt-get install -y -qq cmake g++ git

      - name: Verify GPU access
        run: nvidia-smi

      - name: Cache Kokkos build (CUDA)
        uses: actions/cache@v4
        with:
          path: |
            build/_deps/kokkos-src
            build/_deps/kokkos-build
            build/_deps/kokkos-subbuild
          key: kokkos-cuda-gpu-${{ hashFiles('CMakeLists.txt') }}-${{ runner.os }}
          restore-keys: |
            kokkos-cuda-gpu-${{ hashFiles('CMakeLists.txt') }}-
            kokkos-cuda-gpu-

      - name: Configure CMake (CUDA)
        run: |
          cmake -B ${{ env.BUILD_DIR }} -S . \
            -DCMAKE_CXX_COMPILER=g++ \
            -DCMAKE_BUILD_TYPE=Release \
            -DKokkos_ENABLE_SERIAL=ON \
            -DKokkos_ENABLE_CUDA=ON \
            -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF

      - name: Build
        run: cmake --build ${{ env.BUILD_DIR }}

      - name: Run tests
        run: ctest --test-dir ${{ env.BUILD_DIR }} --output-on-failure --verbose
```

> [!NOTE]
> **No `Kokkos_ARCH_*` flag needed** — when running on real hardware, Kokkos auto-detects the GPU architecture. You only need to hardcode it (like `AMPERE80`) when cross-compiling without a GPU present (your existing `cuda-compile` job).

---

## Step 4: Decide What to Keep

You now have three possible CI jobs. Here's what I'd recommend:

| Job | Runner | Purpose | Keep? |
|-----|--------|---------|-------|
| `host-test` | `ubuntu-latest` | Serial+OpenMP tests | ✅ Yes |
| `cuda-compile` | `ubuntu-latest` + CUDA container | Compile-only check | ⚠️ Optional (redundant if GPU runner is reliable) |
| `cuda-test` | `[self-hosted, gpu]` | Full CUDA build + run | ✅ Yes |

You could keep `cuda-compile` as a fast-fail gate (it runs on GitHub's free runners and catches compile errors without consuming GPU time), or remove it if you want to simplify.

---

## Security Considerations

> [!WARNING]
> **Private repos only.** GitHub strongly advises against using self-hosted runners on **public** repositories. A malicious PR could execute arbitrary code on your machine.

If your repo is public, add a safeguard:

```yaml
  cuda-test:
    # Only run on pushes to your own branches, not on external PRs
    if: github.event_name == 'push' || github.event.pull_request.head.repo.full_name == github.repository
    runs-on: [self-hosted, gpu, linux]
```

---

## Troubleshooting

### Runner connects but jobs hang
```bash
# Check the runner can see the GPU
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### "no CUDA-capable device is detected"
The `--gpus all` flag in the container `options` is missing, or the NVIDIA Container Toolkit isn't installed:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Runner goes offline after reboot
Make sure you installed the systemd service (Step 2, Option B).

### Kokkos fails with wrong architecture
Remove any `Kokkos_ARCH_*` flag — let auto-detection handle it, or set it to match your actual GPU (e.g., `TURING75` for T4, `AMPERE86` for RTX 3060, etc.).

### Check runner version
GitHub enforces minimum runner versions. Update periodically:
```bash
cd ~/actions-runner
./config.sh --check
```
