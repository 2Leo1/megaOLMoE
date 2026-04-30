param(
    [switch]$SkipClone,
    [switch]$SkipBuild,
    [switch]$ForceReclone,
    [switch]$UseCpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK]   $Message" -ForegroundColor Green
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' not found in PATH."
    }
}

function Clone-IfMissing {
    param(
        [string]$RepoUrl,
        [string]$TargetDir,
        [string]$DisplayName
    )

    if (Test-Path $TargetDir) {
        Write-Info "$DisplayName already exists at $TargetDir, skipping clone."
        return
    }

    Write-Info "Cloning $DisplayName..."
    git clone $RepoUrl $TargetDir
}

function Get-HasNvidiaRuntime {
    try {
        $runtimesJson = docker info --format "{{json .Runtimes}}" 2>$null
        if (-not $runtimesJson) {
            return $false
        }
        $runtimes = $runtimesJson | ConvertFrom-Json
        return $null -ne $runtimes.nvidia
    } catch {
        return $false
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "====================================================" -ForegroundColor Blue
Write-Host "     Initializing OLMoE++ Dockerized Workspace      " -ForegroundColor Blue
Write-Host "====================================================" -ForegroundColor Blue

Write-Info "Checking prerequisites..."
Assert-Command "docker"
Assert-Command "git"
docker info *> $null
Write-Ok "Docker and Git are available."

Write-Info "Preparing workspace folders..."
foreach ($dir in @("src", "configs", "data", "logs", "notebooks", "scripts")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot $dir) *> $null
}
foreach ($oldDir in @("venv", ".conda", "miniconda3")) {
    $full = Join-Path $projectRoot $oldDir
    if (Test-Path $full) {
        Remove-Item -Recurse -Force $full
    }
}
Write-Ok "Workspace structure is ready."

$srcRoot = Join-Path $projectRoot "src"
$olmoeDir = Join-Path $srcRoot "OLMoE"
$olmoDir = Join-Path $srcRoot "OLMo"

if ($ForceReclone) {
    Write-WarnMsg "ForceReclone is enabled. Removing OLMo and OLMoE repos before cloning."
    if (Test-Path $olmoeDir) { Remove-Item -Recurse -Force $olmoeDir }
    if (Test-Path $olmoDir) { Remove-Item -Recurse -Force $olmoDir }
}

if (-not $SkipClone) {
    Clone-IfMissing -RepoUrl "https://github.com/allenai/OLMoE.git" -TargetDir $olmoeDir -DisplayName "OLMoE"
    Clone-IfMissing -RepoUrl "https://github.com/allenai/OLMo.git" -TargetDir $olmoDir -DisplayName "OLMo"

    if (Test-Path $olmoDir) {
        Write-Info "Pinning OLMo to the latest commit before 2024-09-01 on main..."
        Push-Location $olmoDir
        try {
            $commit = (git rev-list -n 1 --before="2024-09-01" main).Trim()
            if ($commit) {
                git checkout $commit
                Write-Ok "OLMo pinned to commit $commit"
            } else {
                Write-WarnMsg "Could not find commit before 2024-09-01, keeping current branch state."
            }
        } finally {
            Pop-Location
        }
    }
} else {
    Write-Info "SkipClone is enabled. Reusing existing repositories from src/."
}

Write-Info "Generating Dockerfile..."
$dockerfileContent = @'
# Base image: Official PyTorch Devel (Includes full CUDA toolkit, headers, and GCC)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install -y \
    git ninja-build build-essential vim tmux wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall \
        torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
        "numpy<2.1" \
        "transformers" "wandb" "evaluate" "accelerate" \
        "ninja" "packaging<24.2"

RUN git clone https://github.com/stanford-futuredata/stk.git /workspace/_stk && \
    cd /workspace/_stk && \
    git checkout a1ddf98466730b88a2988860a9d8000fd1833301 && \
    sed -i "s/'torch>=2.3.0,<2.4'/'torch>=2.3.0'/g" setup.py && \
    pip install --no-cache-dir --no-build-isolation .

COPY ./src /workspace/src

WORKDIR /workspace/src/megablocks
RUN MAX_JOBS=4 pip install -e . --no-build-isolation --no-cache-dir --no-deps && \
    SP=$(python -c "import site; print(site.getsitepackages()[0])") && \
    find /workspace/src/megablocks -name "megablocks_ops*.so" -exec cp {} "$SP"/ \;

WORKDIR /workspace/src/OLMo
RUN pip install -e . --no-build-isolation --no-cache-dir --no-deps && \
    pip install --no-cache-dir \
        "omegaconf" "rich" "boto3" "google-cloud-storage" "tokenizers" \
        "ai2-olmo-core==0.1.0" "cached_path" "requests" "torchmetrics" \
        "smashed[remote]>=0.21.1" "safetensors" "datasets" "scikit-learn" \
        "msgspec>=0.14.0" "importlib_resources" || true

WORKDIR /workspace/src/OLMoE
RUN pip install -r requirements.txt --no-cache-dir --no-deps || true

WORKDIR /workspace
CMD ["/bin/bash"]
'@
Set-Content -Path (Join-Path $projectRoot "Dockerfile") -Value $dockerfileContent -NoNewline

$hasNvidiaRuntime = $false
if (-not $UseCpu) {
    $hasNvidiaRuntime = Get-HasNvidiaRuntime
}

Write-Info "Generating docker-compose.yml..."
if ($hasNvidiaRuntime) {
    $composeContent = @'
services:
  olmoe_dev:
    build:
      context: .
      dockerfile: Dockerfile
    image: olmoe-dev-env:latest
    container_name: olmoe_workspace
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./src:/workspace/src
      - ./configs:/workspace/configs
      - ./data:/workspace/data
      - ./logs:/workspace/logs
      - ./scripts:/workspace/scripts
    working_dir: /workspace
    stdin_open: true
    tty: true
    ipc: host
    command: /bin/bash
'@
    Write-Ok "NVIDIA runtime found. GPU-enabled compose file generated."
} else {
    $composeContent = @'
services:
  olmoe_dev:
    build:
      context: .
      dockerfile: Dockerfile
    image: olmoe-dev-env:latest
    container_name: olmoe_workspace
    volumes:
      - ./src:/workspace/src
      - ./configs:/workspace/configs
      - ./data:/workspace/data
      - ./logs:/workspace/logs
      - ./scripts:/workspace/scripts
    working_dir: /workspace
    stdin_open: true
    tty: true
    ipc: host
    command: /bin/bash
'@
    Write-WarnMsg "NVIDIA runtime not detected (or UseCpu enabled). Generated CPU-compatible compose file."
}
Set-Content -Path (Join-Path $projectRoot "docker-compose.yml") -Value $composeContent -NoNewline

Write-Info "Generating Windows helper script start_env.ps1..."
$startEnvContent = @'
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Starting OLMoE++ Docker Environment..." -ForegroundColor Cyan
docker compose run --rm olmoe_dev
'@
Set-Content -Path (Join-Path $projectRoot "start_env.ps1") -Value $startEnvContent -NoNewline

if (-not $SkipBuild) {
    Write-Info "Building Docker image (this is the longest step)..."
    docker compose build
    Write-Ok "Docker image build completed."
} else {
    Write-Info "SkipBuild is enabled. Build step skipped."
}

Write-Host "====================================================" -ForegroundColor Green
Write-Host "        OLMoE++ Architecture Successfully Built!    " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host "To enter your environment from Windows PowerShell:" -ForegroundColor Green
Write-Host "  .\start_env.ps1" -ForegroundColor Blue
Write-Host "====================================================" -ForegroundColor Green