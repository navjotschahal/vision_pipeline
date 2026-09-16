#!/usr/bin/env bash
set -Eeuo pipefail

# Fetch and verify the pinned click-selection research backends (M1).
#
# Creates, next to this repository by default:
#   <models root>/EfficientTAM            yformer/EfficientTAM at a pinned revision
#   <models root>/sam2                    facebookresearch/sam2 at a pinned revision
#   <models root>/model-checkpoints/*.pt  official checkpoints, SHA-256 verified
#
# The repositories are imported from source by the adapter; they are not pip-installed,
# so their optional CUDA extensions are never built into the capture .venv. Only the two
# pure-Python runtime dependencies are added to the .venv. Safe to rerun: anything that
# already matches its pin is left untouched. Override with VISION_SELECTION_* variables.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODELS_ROOT="${VISION_SELECTION_MODELS_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}"
PYTHON="${VISION_SELECTION_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
UV="${REPO_ROOT}/.tools/uv"

EFFICIENTTAM_URL="https://github.com/yformer/EfficientTAM.git"
EFFICIENTTAM_REVISION="abcd061ebd3cc6e7527d152d75b890126aaa53f6"
SAM2_URL="https://github.com/facebookresearch/sam2.git"
SAM2_REVISION="2b90b9f5ceec907a1c18123530e92e794ad901a4"

CHECKPOINTS=(
    "efficienttam_ti.pt|https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_ti.pt|acbb17b28cca1f860acee09c9ecb6efdb732080dc7a85a07292c31813175fa7d"
    "sam2.1_hiera_tiny.pt|https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt|7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
)

checkout() {
    local url="$1" revision="$2" directory="$3"
    if [[ ! -d "${directory}/.git" ]]; then
        echo "Cloning ${url} into ${directory}"
        git clone --quiet "${url}" "${directory}"
    fi
    local current
    current="$(git -C "${directory}" rev-parse HEAD)"
    if [[ "${current}" != "${revision}" ]]; then
        if [[ -n "$(git -C "${directory}" status --porcelain)" ]]; then
            echo "error: ${directory} has local changes; refusing to move it to ${revision}" >&2
            exit 2
        fi
        echo "Checking out ${revision} in ${directory}"
        git -C "${directory}" fetch --quiet origin
        git -C "${directory}" checkout --quiet --detach "${revision}"
    fi
    echo "ok  ${directory} @ ${revision}"
}

verify_checkpoint() {
    local name="$1" url="$2" expected="$3"
    local target="${MODELS_ROOT}/model-checkpoints/${name}"
    if [[ ! -f "${target}" ]]; then
        echo "Downloading ${url}"
        curl -L --fail --retry 3 -o "${target}.part" "${url}"
        mv "${target}.part" "${target}"
    fi
    local actual
    actual="$(sha256sum "${target}" | cut -d' ' -f1)"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "error: ${target} SHA-256 ${actual} does not match pinned ${expected}" >&2
        exit 2
    fi
    echo "ok  ${target} sha256=${expected}"
}

mkdir -p "${MODELS_ROOT}/model-checkpoints"
checkout "${EFFICIENTTAM_URL}" "${EFFICIENTTAM_REVISION}" "${MODELS_ROOT}/EfficientTAM"
checkout "${SAM2_URL}" "${SAM2_REVISION}" "${MODELS_ROOT}/sam2"
for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r name url sha <<<"${entry}"
    verify_checkpoint "${name}" "${url}" "${sha}"
done

if ! "${PYTHON}" -c 'import hydra, iopath' >/dev/null 2>&1; then
    echo "Installing hydra-core and iopath into ${PYTHON}"
    "${UV}" pip install --python "${PYTHON}" "hydra-core==1.3.2" "iopath==0.1.10"
fi
"${PYTHON}" -c 'import hydra, iopath; print("ok  hydra-core", hydra.__version__, "iopath present")'
echo "Selection models ready under ${MODELS_ROOT}"
