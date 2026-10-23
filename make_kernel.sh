#!/bin/bash
# make_kernel.sh
#
# Creates a Jupyter kernel that:
#   - has access to all packages from the given `module load` modules
#   - shows up automatically in Jupyter and VS Code, every time, with no
#     manual setup after this script is run once
#
# Nothing is installed: the kernel simply runs the Python provided by the
# loaded modules. Extra packages are handled case by case.
#
# General usage:
#   ./make_kernel.sh <kernel_name> <module1> [module2 ...]
#
# Advised usage:
#   ./make_kernel.sh idl2026 LIACS/ItDL-2026 [matplotlib]

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <kernel_name> <module1> [module2 ...]"
    exit 1
fi

KERNEL_NAME="$1"
shift
MODULES="$*"

KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"

# Find the module system's init script (Lmod or Environment Modules)
init_modules() {
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "Could not find a module init script — edit this script to point to yours." >&2
        exit 1
    fi
}

echo "==> Loading modules: $MODULES"
init_modules
module load $MODULES

echo "==> Checking that the modules provide ipykernel"
if ! python -c "import ipykernel" 2>/dev/null; then
    echo "ERROR: 'import ipykernel' failed after loading: $MODULES" >&2
    echo "These modules do not provide ipykernel, so a kernel cannot be created from them." >&2
    exit 1
fi

echo "==> Registering persistent kernel '$KERNEL_NAME'"
mkdir -p "$KERNEL_DIR"

# Wrapper script: reloads the same modules every time the kernel actually
# starts, so env vars (not just python packages) are set.
cat > "$KERNEL_DIR/wrapper.sh" << EOF
#!/bin/bash
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi
module load $MODULES
exec python -m ipykernel_launcher -f "\$1"
EOF
chmod +x "$KERNEL_DIR/wrapper.sh"

cat > "$KERNEL_DIR/kernel.json" << EOF
{
  "argv": ["$KERNEL_DIR/wrapper.sh", "{connection_file}"],
  "display_name": "Python ($KERNEL_NAME)",
  "language": "python"
}
EOF

echo ""
echo "Done! Kernel '$KERNEL_NAME' is ready. "
echo "Find it under 'Jupyter Kernel' when selecting a kernel, as 'Python ($KERNEL_NAME)'."
echo "  - It will now always appear in the kernel picker, in any lab computer you log into."
echo "  - Need a package that isn't available in the loaded modules? Ask the lecturer or a TA."
