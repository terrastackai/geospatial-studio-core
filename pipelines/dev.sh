#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


set -e  # Exit on any errors

C3_REPOSITORY="quay.io/geospatial-studio"
DEFAULT_VERSION="v0.1.0"
DEFAULT_OPTIMIZE="true"

# Directory paths
COMPONENTS_DIR="components"
GENERAL_LIBS_DIR="general_libraries"
BUILD_CACHE_FILE=".build_cache.txt"

BUILDABLE_COMPONENTS=(
    "inference-planner"
    "postprocess-generic-single"
    "push_to_geoserver"
    "run-inference"
    "terrakit_data_fetch"
    "terratorch_inference"
    "url-connector-single"
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

log_info() {
    echo -e "${GREEN}[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}" >&2
}

log_debug() {
    echo -e "${BLUE}[DEBUG] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

#Parse the command line arguments i.e. the version passed.
VERSION="${1:-$DEFAULT_VERSION}"
OPTIMIZE_BUILD="${2:-$DEFAULT_OPTIMIZE}"  # NEW: true/false to enable optimize build


# Validate version format (basic check)
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+.*$ ]]; then
    log_warn "Usage: $0 [version] [optimize_build]"
    log_warn "  version        Docker image version (default: $DEFAULT_VERSION)"
    log_warn "  optimize_build Enable build optimization: true/false (default: $DEFAULT_OPTIMIZE)"
    log_warn ""
    log_warn "Examples:"
    log_warn "  $0                    # version: $DEFAULT_VERSION, optimize: $DEFAULT_OPTIMIZE"
    log_warn "  $0 v1.2.3             # version: v1.2.3, optimize: $DEFAULT_OPTIMIZE" 
    log_warn "  $0 v1.2.3 false       # version: v1.2.3, optimize: false"
    log_warn "  $0 v1.2.3 true        # version: v1.2.3, optimize: true"
    exit 1
fi

# Validate optimize_build parameter
if [[ "$OPTIMIZE_BUILD" != "true" && "$OPTIMIZE_BUILD" != "false" ]]; then
    log_error "optimize_build must be 'true' or 'false', got: '$OPTIMIZE_BUILD'"
    log_warn "Usage: $0 [version] [optimize_build]"
    exit 1
fi


calculate_dir_hash() {
    local dir_path="$1"
    if [[ -d "$dir_path" ]]; then
        # More restrictive file inclusion to avoid temp files, IDE files, etc...
        find "$dir_path" -type f \
            \( -name "*.py" -o -name "*.toml" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.sql" -o -name "*.sh" -o -name "Dockerfile*" \) \
            ! -name "*.pyc" \
            ! -name "*.pyo" \
            ! -name "*.pyd" \
            ! -name "*~" \
            ! -name "*.swp" \
            ! -name "*.swo" \
            ! -name ".DS_Store" \
            ! -name "*.tmp" \
            ! -name "*.temp" \
            ! -name "*.log" \
            ! -name "*.pid" \
            ! -name "*.cwl" \
            ! -name "*.job.yaml" \
            ! -name "*.yaml.bak" \
            ! -path "*/.git/*" \
            ! -path "*/__pycache__/*" \
            ! -path "*/.pytest_cache/*" \
            ! -path "*/.coverage/*" \
            ! -path "*/node_modules/*" \
            ! -path "*/.vscode/*" \
            ! -path "*/.idea/*" \
            -exec sha256sum {} \; 2>/dev/null | sort | sha256sum | cut -d' ' -f1
    else
        echo "directory_not_found"
    fi
}

load_build_cache() {
    if [[ -f "$BUILD_CACHE_FILE" ]]; then
        source "$BUILD_CACHE_FILE"
        log_info "Loaded previous build cache"
    else
        log_info "No previous build cache found - this appears to be initial run"
    fi
}


save_build_cache() {
    log_info "Saving build cache..."

    echo "# Build cache generated on $(date)" > "$BUILD_CACHE_FILE"
    echo "PREV_GENERAL_LIBS_HASH=\"$CURRENT_GENERAL_LIBS_HASH\"" >> "$BUILD_CACHE_FILE"
    
    for component in "${BUILDABLE_COMPONENTS[@]}"; do
        local var_name="PREV_${component//-/_}_HASH"
        local current_hash_var="CURRENT_${component//-/_}_HASH"
        echo "${var_name}=\"${!current_hash_var}\"" >> "$BUILD_CACHE_FILE"
    done
}


build_component() {
    local component_name="$1"
    local component_path="${COMPONENTS_DIR}/${component_name}"
    
    log_info "Building component: $component_name"
    
    # Change to the component directory
    cd "$component_path"
    
    if [[ ! -f "Dockerfile.template" ]]; then
        log_warn "No Dockerfile.template found for $component_name - skipping build"
        cd - > /dev/null
        return 0
    fi

    # Clean up any leftover files from previous builds first
    log_debug "Pre-build cleanup for $component_name..."
    rm -rf gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml ./*.yaml 2>/dev/null || true
    
    # Copy general libraries to component directory
    log_info "Copying general libraries to $component_name..."
    cp -r "../../${GENERAL_LIBS_DIR}/gfm_logger" . || { log_error "Failed to copy gfm_logger"; cd - > /dev/null; return 1; }
    cp -r "../../${GENERAL_LIBS_DIR}/gfm_data_processing" . || { log_error "Failed to copy gfm_data_processing"; cd - > /dev/null; return 1; }
    cp -r "../../${GENERAL_LIBS_DIR}/orchestrate_wrapper" . || { log_error "Failed to copy orchestrate_wrapper"; cd - > /dev/null; return 1; }
    
    # Determine main script and additional files based on component
    local main_script="${component_name//-/_}.py"
    local helper_script="${component_name//-/_}_functions.py"
    local additional_files="gfm_logger gfm_data_processing"
    
    # Handle special cases for different components
    case "$component_name" in
        "inference-planner")
            main_script="inference_planner.py"
            helper_script="inference_planner_functions.py"
            additional_files="gfm_logger gfm_data_processing sentinelhub_config.toml"
            ;;
        "postprocess-generic-single")
            main_script="postprocess-generic-single.py"
            helper_script="postprocess_generic_helper_functions.py postprocess_regularization.py"
            ;;
        "push_to_geoserver")
            main_script="push_to_geoserver.py"
            helper_script="push_to_geoserver_helper_functions.py"
            ;;
        "run-inference")
            main_script="run-inference.py"
            helper_script=""  # No helper script for this component
            additional_files="gfm_logger gfm_data_processing inference_helper"
            ;;
        "terrakit_data_fetch")
            main_script="terrakit_data_fetch.py"
            helper_script="" # No helper script for this component
            additional_files="gfm_logger gfm_data_processing sentinelhub_config.toml"
            ;;
        "terratorch_inference")
            main_script="terratorch_inference.py"
            helper_script="terratorch_inference_utils.py"
            ;;
        "url-connector-single")
            main_script="url_connector_single.py"
            helper_script=""
            additional_files="gfm_logger gfm_data_processing preprocessing_helper"
            ;;
    esac
    
    # Build c3 command - this is where we actually build the Docker image using claimed
    local c3_cmd="c3_create_operator --repository ${C3_REPOSITORY} --dockerfile_template_path Dockerfile.template --log_level DEBUG --version ${VERSION} --local_mode ${main_script}"
    
    # Add helper script if it exists
    if [[ ! -z "$helper_script" && -f "$helper_script" ]]; then
        c3_cmd="$c3_cmd $helper_script"
    fi
    
    # Add additional files
    c3_cmd="$c3_cmd $additional_files"
    
    log_info "Executing: $c3_cmd"

    # All occureneces of underscore should be replaced with '-' to match the image name that will be built by claimed.
    local remote_image_name="${C3_REPOSITORY}/claimed-${component_name//_/-}"
    local push_image_cmd="docker push ${remote_image_name}:${VERSION}"
    
    # Execute the c3 command
    if eval "$c3_cmd"; then
        log_info "Successfully built component: $component_name"
        log_info "Pushing the $component_name Image to the Container Registry"
        log_info "Executing: $push_image_cmd"
        if eval "$push_image_cmd"; then
            log_info "Successfully pushed the image for $component_name"
        else
            log_error "Failed to push the image for: $component_name Image is ➡️ ${remote_image_name}:${VERSION}"
            # Cleanup copied libraries and generated files
            log_debug "Post-build cleanup for $component_name..."
            if [[ "$component_name" == "postprocess-generic-single" || "$component_name" == "run-inference" ]]; then
                log_debug "Exception cleanup for ${component_name}"
                rm -rf gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml "${component_name}.yaml" ./*.yaml.bak 2>/dev/null || true
            else
                rm -rf gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml "${component_name//-/_}.yaml" ./*.yaml.bak 2>/dev/null || true
            fi
            cd - > /dev/null
            return 1
        fi
    else
        log_error "Failed to build component: $component_name"
        cd - > /dev/null
        return 1
    fi
    
    # Cleanup copied libraries and generated files
    log_debug "Post-build cleanup for $component_name..."
    if [[ "$component_name" == "postprocess-generic-single" || "$component_name" == "run-inference" ]]; then
        log_debug "Exception cleanup for ${component_name}"
        rm -rf gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml "${component_name}.yaml" ./*.yaml.bak 2>/dev/null || true
    else
        rm -rf gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml "${component_name//-/_}.yaml" ./*.yaml.bak 2>/dev/null || true
    fi
    
    cd - > /dev/null
    return 0
}

# Function to build all components
build_all_components() {
    local reason="$1"
    log_info "Building all components (Reason: $reason, Version: $VERSION)"
    
    local failed_components=()
    for component in "${BUILDABLE_COMPONENTS[@]}"; do
        if ! build_component "$component"; then
            failed_components+=("$component")
        fi
    done
    
    if [[ ${#failed_components[@]} -gt 0 ]]; then
        log_error "Failed to build components: ${failed_components[*]}"
        return 1
    fi
    
    log_info "Successfully built all components"
    return 0
}

# =============================================================================
# MAIN LOGIC
# =============================================================================

main() {
    log_info "Starting smart component build process..."
    log_info "Configuration: Version=$VERSION, Optimize=$OPTIMIZE_BUILD, Repository=$C3_REPOSITORY"
    
    # Setup Tekton results for pipeline compatibility
    # setup_tekton_results
    
    # Check if optimization is enabled
    if [[ "$OPTIMIZE_BUILD" != "true" ]]; then
        log_info "Build optimization disabled - building all components"
        build_all_components "optimization disabled"
        return $?
    fi
    
    log_info "Build optimization enabled - analyzing changes..."
    
    # Load previous build cache
    load_build_cache
    cache_loaded=$?
    log_debug "Cache loaded: $cache_loaded"
    
    # Calculate current hashes
    log_info "Calculating current directory hashes..."
    CURRENT_GENERAL_LIBS_HASH=$(calculate_dir_hash "$GENERAL_LIBS_DIR")
    log_debug "General libraries hash: $CURRENT_GENERAL_LIBS_HASH"
    
    # Calculate hashes for all buildable components
    for component in "${BUILDABLE_COMPONENTS[@]}"; do
        local var_name="CURRENT_${component//-/_}_HASH"
        local component_hash=$(calculate_dir_hash "${COMPONENTS_DIR}/${component}")
        export "$var_name=$component_hash" #make the component hash global to be accessed in save_build_cache
        log_debug "Component $component hash: $component_hash"
    done
    

    # Check if this is the first run or cache was invalid
    if [[ $cache_loaded -ne 0 ]] || [[ -z "$PREV_GENERAL_LIBS_HASH" ]]; then
        log_info "No valid cache found - building all components"
        build_all_components "initial run or invalid cache"
        save_build_cache
        return $?
    fi
    
    # Check if general libraries changed
    if [[ "$CURRENT_GENERAL_LIBS_HASH" != "$PREV_GENERAL_LIBS_HASH" ]]; then
        log_info "Changes detected in general_libraries directory - rebuilding all components"
        log_debug "Previous: $PREV_GENERAL_LIBS_HASH"
        log_debug "Current:  $CURRENT_GENERAL_LIBS_HASH"
        build_all_components "general libraries changed"
        save_build_cache
        return $?
    fi
    
    log_info "No changes in general_libraries - checking individual components..."
    
    # Check individual components for changes
    local changed_components=()
    for component in "${BUILDABLE_COMPONENTS[@]}"; do
        local current_hash_var="CURRENT_${component//-/_}_HASH"
        local prev_hash_var="PREV_${component//-/_}_HASH"
        local current_hash="${!current_hash_var}"
        local prev_hash="${!prev_hash_var}"
        
        if [[ "$current_hash" != "$prev_hash" ]]; then
            log_debug "Previous: $prev_hash"
            log_debug "Current:  $current_hash"
            log_info "Changes detected in component: $component"
            changed_components+=("$component")
        fi
    done
    
    # Build only changed components or all if none changed but we want to be safe
    if [[ ${#changed_components[@]} -eq 0 ]]; then
        log_info "No changes detected in any component - no rebuild necessary"
    else
        log_info "Building ${#changed_components[@]} changed component(s): ${changed_components[*]}"
        local failed_components=()
        for component in "${changed_components[@]}"; do
            if ! build_component "$component"; then
                failed_components+=("$component")
            fi
        done
        
        if [[ ${#failed_components[@]} -gt 0 ]]; then
            log_error "Failed to build changed components: ${failed_components[*]}"
            return 1
        fi
    fi
    
    # Save updated cache
    save_build_cache
    
    log_info "Smart component build process completed successfully"
    return 0
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Display usage if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [version] [optimize_build]"
    echo ""
    echo "Arguments:"
    echo "  version        Docker image version tag (default: $DEFAULT_VERSION)"
    echo "  optimize_build Enable build optimization: true/false (default: $DEFAULT_OPTIMIZE)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Uses defaults: $DEFAULT_VERSION, optimize=$DEFAULT_OPTIMIZE"
    echo "  $0 v1.2.3             # Uses version v1.2.3, optimize=$DEFAULT_OPTIMIZE"
    echo "  $0 v1.2.3 false       # Uses version v1.2.3, no optimization (builds all)"
    echo "  $0 v1.2.3 true        # Uses version v1.2.3, with optimization"
    echo ""
    echo "When to use optimize_build=false:"
    echo "  - First time running after cloning repo"
    echo "  - When you want to force rebuild all components"
    echo "  - When troubleshooting build cache issues"
    echo ""
    exit 0
fi

# Run main function and capture exit code
main
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    log_info "Build process completed successfully"
else
    log_error "Build process failed with exit code: $exit_code"
fi

exit $exit_code