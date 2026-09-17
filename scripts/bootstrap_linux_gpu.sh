#!/usr/bin/env bash
set -Eeuo pipefail

# Build a self-contained Linux/NVIDIA development environment for this repository.
# Configuration is through VISION_* environment variables; no system Python packages
# or shell startup files are changed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/.tools"
VENV_DIR="${REPO_ROOT}/.venv"

UV_VERSION="${VISION_UV_VERSION:-0.12.14}"
PYTHON_VERSION="${VISION_PYTHON_VERSION:-3.12}"
TORCH_VERSION="${VISION_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${VISION_TORCHVISION_VERSION:-0.22.1}"
TORCH_INDEX_URL="${VISION_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
REALSENSE_VERSION="${VISION_REALSENSE_VERSION:-2.56.5.9235}"
REQUIRE_CUDA="${VISION_REQUIRE_CUDA:-1}"
REQUIRE_REALSENSE="${VISION_REQUIRE_REALSENSE:-1}"

# Large CUDA wheels can take several minutes on the lab network. uv's default HTTP
# timeout is intentionally short for ordinary packages, so give these transfers room
# to complete while retaining uv's normal retry behavior.
export UV_HTTP_TIMEOUT="${VISION_HTTP_TIMEOUT:-300}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: this bootstrap is for Linux; detected $(uname -s)" >&2
    exit 2
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "error: the pinned CUDA/RealSense wheels require Linux x86_64" >&2
    exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required to install uv" >&2
    exit 2
fi
if [[ "${REQUIRE_CUDA}" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi is unavailable (set VISION_REQUIRE_CUDA=0 to bypass)" >&2
    exit 2
fi

mkdir -p "${TOOLS_DIR}"
UV="${TOOLS_DIR}/uv"
if [[ ! -x "${UV}" ]]; then
    echo "Installing uv ${UV_VERSION} in ${TOOLS_DIR}"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        | env UV_UNMANAGED_INSTALL="${TOOLS_DIR}" sh
fi

echo "Installing managed Python ${PYTHON_VERSION}"
"${UV}" python install "${PYTHON_VERSION}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    EXISTING_VERSION="$(${VENV_DIR}/bin/python -c 'import platform; print(platform.python_version())')"
    case "${EXISTING_VERSION}" in
        "${PYTHON_VERSION}"|"${PYTHON_VERSION}".*) ;;
        *)
            echo "error: ${VENV_DIR} uses Python ${EXISTING_VERSION}, expected ${PYTHON_VERSION}." >&2
            echo "Move the existing .venv aside and rerun the script." >&2
            exit 2
            ;;
    esac
else
    "${UV}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"

echo "Installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
"${UV}" pip install --python "${PYTHON}" --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

echo "Installing vision-pipeline and host dependencies"
"${UV}" pip install --python "${PYTHON}" \
    "pyrealsense2==${REALSENSE_VERSION}" \
    --editable "${REPO_ROOT}[webcam,yolo,dino,fusion,dev]"

echo "Verifying Python, CUDA, project imports, and RealSense discovery"
VISION_REQUIRE_CUDA="${REQUIRE_CUDA}" \
VISION_REQUIRE_REALSENSE="${REQUIRE_REALSENSE}" \
"${PYTHON}" - <<'PY'
import os

import cv2
import pyrealsense2 as rs
import torch
import vision_pipeline

require_cuda = os.environ["VISION_REQUIRE_CUDA"] == "1"
require_realsense = os.environ["VISION_REQUIRE_REALSENSE"] == "1"

print(f"vision_pipeline={vision_pipeline.__file__}")
print(f"opencv={cv2.__version__}")
print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("CUDA verification failed")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    value = (torch.ones(1024, device=device) * 2).sum().item()
    print(f"cuda_device={torch.cuda.get_device_name(0)} tensor_check={value:.0f}")

devices = list(rs.context().query_devices())
print(f"realsense_devices={len(devices)}")
for device in devices:
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    firmware = device.get_info(rs.camera_info.firmware_version)
    print(f"realsense name={name!r} serial={serial} firmware={firmware}")
if require_realsense and not devices:
    raise SystemExit("RealSense verification failed: no device is visible")
PY

echo
echo "Environment ready. Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
