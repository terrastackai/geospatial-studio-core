#!/usr/bin/env python3
"""
© Copyright IBM Corporation 2026
SPDX-License-Identifier: Apache-2.0

Automated Model Deployment Script
This script demonstrates how to deploy models using the vllm-inference-server template.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def load_template(template_path: str) -> str:
    """Load the YAML template file."""
    with open(template_path, "r") as f:
        return f.read()


def replace_placeholders(template: str, config: Dict[str, str]) -> str:
    """Replace all placeholders in the template with actual values."""
    result = template
    for key, value in config.items():
        placeholder = f"${{{key}}}"
        result = result.replace(placeholder, str(value))
    return result


def filter_by_kserve_mode(yaml_content: str, enable_kserve: bool) -> str:
    """
    Filter the YAML content based on KServe mode.
    Remove Deployment if KServe is enabled, or remove InferenceService if disabled.
    """
    lines = yaml_content.split("\n")
    filtered_lines = []
    skip_section = False
    current_section = None

    for line in lines:
        # Detect section starts
        if line.startswith("# Deployment (used when ENABLE_KSERVE=false)"):
            current_section = "deployment"
            skip_section = enable_kserve  # Skip if KServe is enabled
        elif line.startswith("# InferenceService (used when ENABLE_KSERVE=true)"):
            current_section = "inferenceservice"
            skip_section = not enable_kserve  # Skip if KServe is disabled
        elif line.startswith("---") and current_section:
            # End of section
            current_section = None
            skip_section = False

        # Add line if not skipping
        if not skip_section:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def deploy_model(
    model_name: str,
    namespace: str,
    enable_kserve: bool = False,
    image_repository: str = "us.icr.io/gfmaas/vllm-small",
    image_tag: str = "v0.0.6",
    image_pull_secret: str = "my-registry-secret",
    service_account: str = "default",
    models_pvc: str = "vllm-models-pvc",
    inference_shared_pvc: str = "inference-shared-pvc",
    gpu_count: str = "1",
    cpu_limit: str = "2000m",
    memory_limit: str = "8Gi",
    cpu_request: str = "1000m",
    memory_request: str = "4Gi",
    dry_run: bool = False,
    output_file: Optional[str] = None,
) -> bool:
    """
    Deploy a model using the template.

    Args:
        model_name: Name of the model (will be prefixed with gfm-amo-)
        namespace: Kubernetes namespace
        enable_kserve: Whether to use KServe InferenceService
        image_repository: Container image repository
        image_tag: Container image tag
        image_pull_secret: Name of image pull secret
        service_account: Service account name
        models_pvc: PVC name for models storage
        inference_shared_pvc: PVC name for shared inference data
        gpu_count: Number of GPUs to request
        cpu_limit: CPU limit
        memory_limit: Memory limit
        cpu_request: CPU request
        memory_request: Memory request
        dry_run: If True, only generate YAML without applying
        output_file: If provided, save generated YAML to this file

    Returns:
        True if successful, False otherwise
    """
    # Configuration dictionary
    config = {
        "MODEL_NAME": model_name,
        "NAMESPACE": namespace,
        "ENABLE_KSERVE": str(enable_kserve).lower(),
        "IMAGE_REPOSITORY": image_repository,
        "IMAGE_TAG": image_tag,
        "IMAGE_PULL_SECRET": image_pull_secret,
        "SERVICE_ACCOUNT": service_account,
        "MODELS_PVC": models_pvc,
        "INFERENCE_SHARED_PVC": inference_shared_pvc,
        "GPU_COUNT": gpu_count,
        "CPU_LIMIT": cpu_limit,
        "MEMORY_LIMIT": memory_limit,
        "CPU_REQUEST": cpu_request,
        "MEMORY_REQUEST": memory_request,
    }

    # Load template
    template_path = Path(__file__).parent / "vllm-inference-server-template.yaml"
    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}", file=sys.stderr)
        return False

    template = load_template(str(template_path))

    # Replace placeholders
    yaml_content = replace_placeholders(template, config)

    # Filter based on KServe mode
    yaml_content = filter_by_kserve_mode(yaml_content, enable_kserve)

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(yaml_content)
        print(f"Generated YAML saved to: {output_file}")

    # Print or apply
    if dry_run:
        print("=" * 80)
        print("DRY RUN - Generated YAML:")
        print("=" * 80)
        print(yaml_content)
        print("=" * 80)
        deployment_type = (
            "KServe InferenceService" if enable_kserve else "Standard Deployment"
        )
        print(f"\nDeployment type: {deployment_type}")
        print(f"Service name: gfm-amo-{model_name}")
        print(f"Namespace: {namespace}")
        return True
    else:
        # Apply to Kubernetes
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=yaml_content.encode(),
                capture_output=True,
                check=True,
            )
            print(result.stdout.decode())
            deployment_type = (
                "KServe InferenceService" if enable_kserve else "Standard Deployment"
            )
            print(
                f"\n✓ Successfully deployed {deployment_type} for model: gfm-amo-{model_name}"
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error deploying model: {e.stderr.decode()}", file=sys.stderr)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy GFM inference server for a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy with standard Deployment
  python deploy_model.py my-model --namespace geospatial-studio --dry-run
  
  # Deploy with KServe InferenceService
  python deploy_model.py my-model --namespace geospatial-studio --enable-kserve --dry-run
  
  # Deploy with custom resources
  python deploy_model.py my-model --namespace geospatial-studio \\
    --gpu-count 2 --memory-limit 16Gi --cpu-limit 4000m
  
  # Generate YAML file without deploying
  python deploy_model.py my-model --namespace geospatial-studio \\
    --dry-run --output my-model-deployment.yaml
        """,
    )

    parser.add_argument("model_name", help="Name of the model to deploy")
    parser.add_argument("--namespace", required=True, help="Kubernetes namespace")
    parser.add_argument(
        "--enable-kserve",
        action="store_true",
        help="Use KServe InferenceService instead of standard Deployment",
    )
    parser.add_argument(
        "--image-repository",
        default="us.icr.io/gfmaas/vllm-small",
        help="Container image repository",
    )
    parser.add_argument("--image-tag", default="v0.0.6", help="Container image tag")
    parser.add_argument(
        "--image-pull-secret",
        default="my-registry-secret",
        help="Name of image pull secret",
    )
    parser.add_argument(
        "--service-account", default="default", help="Service account name"
    )
    parser.add_argument(
        "--models-pvc", default="vllm-models-pvc", help="PVC name for models storage"
    )
    parser.add_argument(
        "--inference-shared-pvc",
        default="inference-shared-pvc",
        help="PVC name for shared inference data",
    )
    parser.add_argument("--gpu-count", default="1", help="Number of GPUs to request")
    parser.add_argument("--cpu-limit", default="2000m", help="CPU limit")
    parser.add_argument("--memory-limit", default="8Gi", help="Memory limit")
    parser.add_argument("--cpu-request", default="1000m", help="CPU request")
    parser.add_argument("--memory-request", default="4Gi", help="Memory request")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate YAML without applying to cluster",
    )
    parser.add_argument("--output", help="Save generated YAML to file")

    args = parser.parse_args()

    success = deploy_model(
        model_name=args.model_name,
        namespace=args.namespace,
        enable_kserve=args.enable_kserve,
        image_repository=args.image_repository,
        image_tag=args.image_tag,
        image_pull_secret=args.image_pull_secret,
        service_account=args.service_account,
        models_pvc=args.models_pvc,
        inference_shared_pvc=args.inference_shared_pvc,
        gpu_count=args.gpu_count,
        cpu_limit=args.cpu_limit,
        memory_limit=args.memory_limit,
        cpu_request=args.cpu_request,
        memory_request=args.memory_request,
        dry_run=args.dry_run,
        output_file=args.output,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
