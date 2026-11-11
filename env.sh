# Shell environment for the Mini_Dice compiler.
#
#   source env.sh
#
# Submodules under third_party/ are the source pins. The LLVM plugin and the
# VEGAS conda toolchain are build products and are not in those repositories.
# This file uses a toolchain that already exists next to this checkout when
# the submodule itself has not been built. Place and route runs in the same
# Python as the flow; it does not need the CGRA-Solve virtualenv.

if [ -z "${BASH_VERSION:-}${ZSH_VERSION:-}" ]; then
    echo "minidice-env: source this file from bash or zsh" >&2
    return 1 2>/dev/null || exit 1
fi
if [ -n "${BASH_VERSION:-}" ]; then
    case "${BASH_SOURCE[0]}" in
        "$0") echo "minidice-env: source this file, do not execute it" >&2; exit 1 ;;
    esac
    _MD_ENV_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _MD_ENV_FILE="${(%):-%N}"
fi

_MD_ROOT="$(cd "$(dirname "$_MD_ENV_FILE")" && pwd)"
export MINIDICE_ROOT="$_MD_ROOT"
export BOXMAN_ROOT="$_MD_ROOT/third_party/boxman"

# Clang and the two Python environments are built outside this repo. Use one
# the caller named, then the VEGAS toolchain this flow was developed against.
_md_tools=""
for _cand in \
    "${VEGAS_TOOLS_ROOT:-}" \
    "/data2/${USER:-}/VEGAS/build/vegas-rtl-tools-local/envs"
do
    if [ -n "$_cand" ] && [ -x "$_cand/compiler/bin/python3.10" ]; then
        _md_tools="$_cand"
        break
    fi
done
if [ -z "$_md_tools" ]; then
    echo "minidice-env: no VEGAS toolchain; set VEGAS_TOOLS_ROOT" >&2
else
    export MINIDICE_COMPILER_PYTHON="$_md_tools/compiler/bin/python3.10"
    export MINIDICE_DORA_PYTHON="$_md_tools/dora/bin/python3.12"
    export BOXMAN_TOOLCHAIN_BIN="$_md_tools/compiler/bin"
    case ":$PATH:" in
        *":$BOXMAN_TOOLCHAIN_BIN:"*) ;;
        *) export PATH="$BOXMAN_TOOLCHAIN_BIN:$PATH" ;;
    esac
fi

# The CUDA frontend needs a toolkit. Use one the caller named, else the
# toolkit this flow was developed against.
if [ -z "${CUDA_HOME:-}${CUDA_PATH:-}${CUDA_INSTALL_PATH:-}" ]; then
    if [ -x /usr/local/cuda-11.7/bin/nvcc ]; then
        export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
        export CUDA_HOME="$CUDA_INSTALL_PATH"
        export CUDA_PATH="$CUDA_INSTALL_PATH"
        case ":$PATH:" in
            *":$CUDA_INSTALL_PATH/bin:"*) ;;
            *) export PATH="$CUDA_INSTALL_PATH/bin:$PATH" ;;
        esac
    fi
fi

# The frontend loads libBoxmanFrontend.so from inside the boxman tree.
_md_plugin="$BOXMAN_ROOT/llvm/build/libBoxmanFrontend.so"
if [ ! -e "$BOXMAN_ROOT/llvm/build" ]; then
    for _donor in \
        "${BOXMAN_BUILD:-}" \
        "/data2/${USER:-}/VEGAS/compiler/boxman/llvm/build"
    do
        if [ -n "$_donor" ] && [ -f "$_donor/libBoxmanFrontend.so" ]; then
            ln -s "$_donor" "$BOXMAN_ROOT/llvm/build"
            echo "minidice-env: linked boxman plugin build from $_donor"
            break
        fi
    done
fi
if [ ! -f "$_md_plugin" ]; then
    echo "minidice-env: Boxman LLVM plugin is not built at $_md_plugin" >&2
fi

echo "minidice-env: MINIDICE_ROOT=$MINIDICE_ROOT"
echo "minidice-env: compiler python=${MINIDICE_COMPILER_PYTHON:-unset}"
echo "minidice-env: dora python=${MINIDICE_DORA_PYTHON:-unset}"
echo "minidice-env: CUDA_HOME=${CUDA_HOME:-unset}"

unset _MD_ENV_FILE _MD_ROOT _md_tools _md_plugin _cand _donor
