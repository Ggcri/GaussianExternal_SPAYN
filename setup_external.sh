#!/bin/bash
#
# Setup script for External Quantum Chemistry Interface
# This script generates a module file that configures the working environment.
# Environment variables point directly to the ExtScript/ subdirectories
# within this repository.
#
# Usage: ./setup_external.sh [--python PATH] [--module-name NAME]
#
# The --python option sets EXT_PYTHON_PATH in the module file.
# CentralExt uses this variable to find the correct Python interpreter.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration variables
PYTHON_PATH=""
MODULE_NAME="external_tools"
MODULE_NAME_SET=false
SKIP_PYTHON=false

# ============================================================================
# Utility functions
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}  External Quantum Chemistry Interface - Setup Script${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# ============================================================================
# CLI argument parsing
# ============================================================================

parse_args() {
    while [ $# -gt 0 ]; do
        case $1 in
            --python)
                if [ -z "$2" ] || echo "$2" | grep -q '^--'; then
                    print_error "--python requires a path argument"
                    exit 1
                fi
                PYTHON_PATH="$2"
                shift 2
                ;;
            --module-name)
                if [ -z "$2" ] || echo "$2" | grep -q '^--'; then
                    print_error "--module-name requires a name argument"
                    exit 1
                fi
                MODULE_NAME="$2"
                MODULE_NAME_SET=true
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

set_defaults() {
    if [ -n "$PYTHON_PATH" ]; then
        SKIP_PYTHON=false
        print_info "EXT_PYTHON_PATH will be set to: $PYTHON_PATH"
    else
        SKIP_PYTHON=true
        print_info "EXT_PYTHON_PATH will NOT be set (no --python provided)"
        print_info "CentralExt will use 'python3' from PATH by default"
    fi
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo -e "${BLUE}OPTIONS (all optional):${NC}"
    echo "  --python PATH      Path to Python 3.9+ interpreter (sets EXT_PYTHON_PATH)"
    echo "  --module-name NAME Name of the module file (default: external_tools)"
    echo "  -h, --help         Show this message"
    echo ""
    echo -e "${BLUE}BEHAVIOR:${NC}"
    echo "  - If --python is provided: EXT_PYTHON_PATH is set in the module file"
    echo "  - If --python is NOT provided: CentralExt uses 'python3' from PATH"
    echo "  - Environment variables point directly to ExtScript/ subdirectories"
    echo "  - Module file is generated in the script directory"
    echo ""
    echo -e "${BLUE}HOW TO FIND PYTHON PATH:${NC}"
    echo "  which python3.12 python3.11 python3.10 python3.9 2>/dev/null | head -1"
    echo "  module avail python 2>&1 | grep -i python"
    echo ""
    echo -e "${BLUE}EXAMPLES:${NC}"
    echo ""
    echo "  Minimal (uses python3 from PATH):"
    echo "    $0"
    echo ""
    echo "  Set specific Python interpreter:"
    echo "    $0 --python /usr/bin/python3.12"
    echo ""
    echo "  Custom module name:"
    echo "    $0 --python /usr/bin/python3.12 --module-name mymodule"
}

# ============================================================================
# Interactive prompts
# ============================================================================

prompt_user() {
    if [ "$MODULE_NAME_SET" = false ]; then
        echo ""
        echo -e "${BLUE}[OPTIONAL] Module file name [default: $MODULE_NAME]:${NC}"
        echo "(Press Enter to use the default)"
        read -r -p "> " input_module_name
        if [ -n "$input_module_name" ]; then
            MODULE_NAME="$input_module_name"
        fi
    fi
}

# ============================================================================
# Python validation
# ============================================================================

validate_python() {
    print_info "Validating Python interpreter..."

    # Expand ~ if present
    case "$PYTHON_PATH" in
        ~*) PYTHON_PATH="$HOME${PYTHON_PATH#\~}" ;;
    esac

    # Check existence
    if [ ! -x "$PYTHON_PATH" ]; then
        print_error "Python not found or not executable: $PYTHON_PATH"
        exit 1
    fi

    # Check version (POSIX-compatible: works on both Linux and macOS)
    version=$("$PYTHON_PATH" --version 2>&1 | sed 's/Python \([0-9]*\.[0-9]*\).*/\1/')
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)

    if [ "$major" -lt 3 ] 2>/dev/null || { [ "$major" -eq 3 ] && [ "$minor" -lt 9 ]; }; then
        print_error "Python version $version not supported. Requires 3.9+"
        exit 1
    fi

    print_success "Python $version found: $PYTHON_PATH"
}

# ============================================================================
# Validate ExtScript directory
# ============================================================================

validate_extscript() {
    print_info "Checking ExtScript directory..."

    local extscript_dir="$SCRIPT_DIR/ExtScript"
    if [ ! -d "$extscript_dir" ]; then
        print_error "ExtScript directory not found: $extscript_dir"
        print_error "This script must be run from the repository root."
        exit 1
    fi

    # Count available template files
    local count=0
    for subdir in EndingMolpro PreambleMolpro PreambleGau PreambleMR EndingMR \
                  Gaussian Molpro MRCC Orca BasisSet DisplacementStrategies WorkflowTemplate; do
        if [ -d "$extscript_dir/$subdir" ]; then
            count=$((count + 1))
        fi
    done

    print_success "ExtScript directory found with $count subdirectories"

    # Set executable permissions on workflow tools
    local workflow_script="$extscript_dir/WorkflowTemplate/create_workflow.py"
    if [ -f "$workflow_script" ]; then
        chmod +x "$workflow_script"
        print_success "Set executable permissions on create_workflow.py"
    fi
}

# ============================================================================
# Generate module file
# ============================================================================

generate_module() {
    print_info "Generating module file..."

    local module_file="$SCRIPT_DIR/${MODULE_NAME}.module"
    local extscript_dir="$SCRIPT_DIR/ExtScript"

    # Prepare EXT_PYTHON_PATH line (only if Python was provided)
    local python_env_line=""
    local python_help_line=""
    if [ "$SKIP_PYTHON" = false ]; then
        python_env_line="
# Python interpreter for CentralExt wrapper
setenv EXT_PYTHON_PATH \"$PYTHON_PATH\""
        python_help_line="
    puts stderr \"- Python interpreter (EXT_PYTHON_PATH)\""
    fi

    cat > "$module_file" << EOF
#%Module1.0
##
## Module file for External Quantum Chemistry Interface
## Automatically generated by setup_external.sh
## Date: $(date '+%Y-%m-%d %H:%M:%S')
##

proc ModulesHelp { } {
    puts stderr "This module loads environment variables for:"
    puts stderr "- External executables (ELECEXT_PATH)"${python_help_line}
    puts stderr "- Molpro scripts (EMOL, PMOL)"
    puts stderr "- Gaussian scripts (PGAU, EGAU)"
    puts stderr "- MRCC scripts (EMRCC, PMRCC)"
    puts stderr "- Orca scripts (EORCA, PORCA)"
    puts stderr "- Basis sets (EBAS)"
    puts stderr ""
    puts stderr "NOTE: Set SCRATCH or TMPDIR in your PBS/SLURM job script"
}

module-whatis "Environment variables for External QC Interface"

# Base path to external executables
setenv ELECEXT_PATH "$SCRIPT_DIR/Executables"

# Add executables path to PATH
prepend-path PATH "$SCRIPT_DIR/Executables"${python_env_line}

# Molpro scripts
setenv PMOL "$extscript_dir/PreambleMolpro"
setenv EMOL "$extscript_dir/EndingMolpro"

# Gaussian scripts
setenv PGAU "$extscript_dir/PreambleGau"
setenv EGAU "$extscript_dir/Gaussian/Ending"

# MRCC scripts
setenv PMRCC "$extscript_dir/PreambleMR"
setenv EMRCC "$extscript_dir/EndingMR"

# Orca scripts
setenv PORCA "$extscript_dir/Orca/Preamble"
setenv EORCA "$extscript_dir/Orca/Ending"

# Basis sets
setenv EBAS "$extscript_dir/BasisSet"

# Workflow tools (create_workflow.py)
prepend-path PATH "$extscript_dir/WorkflowTemplate"

# ==========================================================================
# WARNING: SCRATCH and TMPDIR
# ==========================================================================
# These variables should be set in your PBS/SLURM job script or shell
# profile. The external interface uses SCRATCH (with TMPDIR as fallback)
# for temporary files during calculations.
#
# Uncomment and modify ONLY if you need to set default values:
#
# setenv SCRATCH "/path/to/scratch"
# setenv TMPDIR "/path/to/scratch"
# ==========================================================================
EOF

    print_success "Module file generated: $module_file"
}

# ============================================================================
# Print summary
# ============================================================================

print_summary() {
    local extscript_dir="$SCRIPT_DIR/ExtScript"

    echo ""
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}  Setup completed successfully!${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo ""

    echo -e "${BLUE}Configuration summary:${NC}"
    if [ "$SKIP_PYTHON" = true ]; then
        echo "  Python:     (not set - CentralExt will use 'python3' from PATH)"
    else
        echo "  Python:     $PYTHON_PATH"
    fi
    echo "  Module:     $SCRIPT_DIR/${MODULE_NAME}.module"
    echo ""
    echo -e "${BLUE}Environment variables in module file:${NC}"
    echo "  ELECEXT_PATH     -> $SCRIPT_DIR/Executables"
    echo "  PATH             -> includes $SCRIPT_DIR/Executables"
    if [ "$SKIP_PYTHON" = false ]; then
        echo "  EXT_PYTHON_PATH  -> $PYTHON_PATH"
    fi
    echo "  PMOL             -> $extscript_dir/PreambleMolpro"
    echo "  EMOL             -> $extscript_dir/EndingMolpro"
    echo "  PGAU             -> $extscript_dir/PreambleGau"
    echo "  EGAU             -> $extscript_dir/Gaussian/Ending"
    echo "  PMRCC            -> $extscript_dir/PreambleMR"
    echo "  EMRCC            -> $extscript_dir/EndingMR"
    echo "  PORCA            -> $extscript_dir/Orca/Preamble"
    echo "  EORCA            -> $extscript_dir/Orca/Ending"
    echo "  EBAS             -> $extscript_dir/BasisSet"
    echo "  PATH             -> includes $extscript_dir/WorkflowTemplate"
    echo ""
    echo -e "${YELLOW}Workflow tool:${NC}"
    echo "  After loading the module, create a DPCS3+PCS2 workflow from an XYZ file:"
    echo "    create_workflow.py molecule.xyz --help"
    echo ""
    echo -e "${YELLOW}How to use the module file:${NC}"
    echo ""
    echo "  Option 1 - Direct load:"
    echo "    module load $SCRIPT_DIR/${MODULE_NAME}.module"
    echo ""
    echo "  Option 2 - Add to MODULEPATH:"
    echo "    module use $SCRIPT_DIR"
    echo "    module load $MODULE_NAME"
    echo ""
    echo "  Option 3 - Copy to modulefiles directory:"
    echo "    cp $SCRIPT_DIR/${MODULE_NAME}.module \$MODULEPATH/${MODULE_NAME}"
    echo "    module load $MODULE_NAME"
    echo ""
    echo -e "${YELLOW}IMPORTANT:${NC} Before running calculations, ensure:"
    echo "  1. Gaussian is loaded/available in your environment"
    echo "  2. The external QC program (Molpro, ORCA, MRCC) is loaded"
    echo "  3. SCRATCH or TMPDIR is set to a local scratch directory"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    print_header

    # Parse command line arguments
    parse_args "$@"

    # Set defaults for missing arguments
    set_defaults

    # Request optional input interactively (only module name)
    prompt_user

    # Validate Python if provided
    if [ "$SKIP_PYTHON" = false ]; then
        validate_python
    fi

    # Validate ExtScript directory
    validate_extscript

    # Generate module file
    generate_module

    # Summary
    print_summary
}

# Run main with all arguments
main "$@"
