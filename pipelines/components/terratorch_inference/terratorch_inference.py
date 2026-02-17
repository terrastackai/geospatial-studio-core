# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
This starter component script
"""

# Dependencies
# pip install ibm-cos-sdk requests tenacity pyyaml opentelemetry-distro opentelemetry-exporter-otlp

import os
import shutil
import json
import glob
import yaml
import time


from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.metrics import MetricManager
from terratorch_inference_utils import (
    read_json_with_retries,
    copy_tiffs,
    delete_tmp_dir,
)

# Uncomment next 2 lines for local testing
# import dotenv
# dotenv.load_dotenv()

# inference folder
inference_folder = os.environ.get("inference_folder", "")

tunes_folder = os.getenv("tunes_folder", "/tunes/tune-tasks")

# inference_id
inference_id = os.environ.get("inference_id", "test-inference-1")

# task_id
task_id = os.environ.get("task_id", f"{inference_id}-task_0")

process_id = os.getenv("process_id", "terratorch-inference")

metric_manager = MetricManager(component_name=process_id)


@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
def run_terratorch_inference():
    try:
        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=task_id,
            event_status="Inferencing ..",
        )
        ######################################################################################################
        ###  Parse the inference and task configs from file
        ######################################################################################################

        inference_config_path = f"{inference_folder}/{inference_id}_config.json"

        with open(inference_config_path, "r") as fp:
            inference_dict = json.load(fp)

        task_folder = f"{inference_folder}/{task_id}"

        task_config_path = f"{task_folder}/{task_id}_config.json"
        task_dict = read_json_with_retries(filepath=task_config_path, max_retries=5)

        # logger.info(f"Loaded inference and task configs")

        ######################################################################################################
        ###  Add your processing code here
        ######################################################################################################

        tune_id = inference_dict["tune_id"]
        tune_path = f"{tunes_folder}/{tune_id}".replace("//", "/")

        model_config_path = f"{tune_path}/config_deploy.yaml"
        model_checkpoint_path = glob.glob(f"{tune_path}/*state*.ckpt")[0]
        output_folder = task_folder

        logger.info(f"Tune path: {tune_path}")

        with open(model_config_path, "r") as fp:
            config = yaml.safe_load(fp)

        ## TODO: Check if tiled_inference is turned on. If not, do not this bit,
        ## TODO: If turned on, have default values but allow user to pass specific values.
        ## Check the config to see if it has tiled inference, otherwise add the block
        if "tiled_inference_parameters" not in config["model"]["init_args"]:
            config["model"]["init_args"]["tiled_inference_parameters"] = {
                "h_crop": 224,
                "h_stride": 208,
                "w_crop": 224,
                "w_stride": 208,
                "batch_size": 16,
                "average_patches": True,
                "verbose": False,
            }
            # logger.info(f"Added tiled inference")

        # write the updated config to a new config file
        class MyDumper(yaml.Dumper):
            def increase_indent(self, flow=False, indentless=False):
                return super(MyDumper, self).increase_indent(flow, False)

        with open(f"{tune_path}/config_deploy_tiled.yaml", "w") as fp:
            fp.write(yaml.dump(config, Dumper=MyDumper, default_flow_style=False))

        ## Copy the tiffs to a new tmp folder, remove modality tags from the names
        model_config_path = f"{tune_path}/config_deploy_tiled.yaml"
        tmp_folder = os.path.join(task_folder, "tmp-inf")
        tmp_regression_images_dir = os.path.join(task_folder, "regression_tmp")

        # input data spec will be the task folder for uni-modal, and a dict if multi-modal (need check)
        if "modalities" in config["data"]["init_args"]:
            modality_tags = config["data"]["init_args"]["modalities"]
            ## Create a tmp folder, rename the tiffs to remove modality tag
            os.makedirs(tmp_folder, exist_ok=True)  # safer and won't error if exists
            for filename in os.listdir(task_folder):
                for modality_tag in modality_tags:
                    if (
                        "imputed" in filename.lower()
                        and filename.lower().endswith((".tif", ".tiff"))
                        and modality_tag in filename
                    ):
                        # Split filename by "_" and remove modality_tag
                        parts = filename.split("_")
                        new_parts = [part for part in parts if part != modality_tag]
                        new_filename = "_".join(new_parts)

                        # Build full paths
                        old_path = os.path.join(task_folder, filename)
                        new_path = os.path.join(tmp_folder, new_filename)

                        # Copy the file
                        print(f"Copying {old_path} to {new_path}")
                        shutil.copy2(old_path, new_path)

            # make dict here
            temp_spec = {}
            for i in config["data"]["init_args"]["modalities"]:
                temp_spec[i] = f"{output_folder}/tmp-inf/"

            input_data_spec = json.dumps(temp_spec)

            terratorch_cli_command = f'terratorch predict -c "{model_config_path}" --ckpt_path "{model_checkpoint_path}" --predict_output_dir {output_folder} --data.init_args.predict_data_root "{input_data_spec}"'

            ## TODO: Terramind not happy with image_grep command, remove it for now. (Fix later)
            # for i in config["data"]["init_args"]["modalities"]:
            #     img_grep = [X for X in task_dict["imputed_input_image"] if i in X][0]

            #     terratorch_cli_command += f' --data.init_args.img_grep.{i} "{img_grep}"'
        # Regression needs the prediction files to be in a separate folder.
        elif (
            "GenericNonGeoPixelwiseRegressionDataModule" == config["data"]["class_path"]
        ):
            # Remove tiled_inference for the regression task.
            with open(model_config_path, "r") as fp:
                config_tiled = yaml.safe_load(fp)
            config_tiled["model"]["init_args"].pop("tiled_inference_parameters", None)

            with open(f"{tune_path}/config_deploy_not_tiled.yaml", "w") as fp:
                fp.write(
                    yaml.dump(config_tiled, Dumper=MyDumper, default_flow_style=False)
                )

            # Create tmp dirs and move imgs there to run inference.
            os.makedirs(tmp_regression_images_dir, exist_ok=True)
            copy_tiffs(
                search_key="imputed",
                old_path=task_folder,
                new_path=tmp_regression_images_dir,
            )
            model_config_path_not_tiled = (
                f"{tune_path}/config_deploy_not_tiled.yaml"
            )
            input_data_spec = tmp_regression_images_dir
            terratorch_cli_command = f'terratorch predict -c "{model_config_path_not_tiled}" --ckpt_path "{model_checkpoint_path}" --predict_output_dir {output_folder} --data.init_args.predict_data_root {input_data_spec}'

        else:
            input_data_spec = output_folder
            if isinstance(task_dict["imputed_input_image"], list):
                img_grep = task_dict["imputed_input_image"][0]
            elif isinstance(task_dict["imputed_input_image"], str):
                img_grep = task_dict["imputed_input_image"]
            terratorch_cli_command = f'terratorch predict -c "{model_config_path}" --ckpt_path "{model_checkpoint_path}" --predict_output_dir {output_folder} --data.init_args.predict_data_root {input_data_spec} --data.init_args.img_grep {img_grep}'

        ## Now run the command and get a list of the inference output tifs

        print(terratorch_cli_command)

        os.system(terratorch_cli_command)
        os.system("sync")

        # delete the Tmp dirs for images
        delete_tmp_dir(tmp_folder)
        delete_tmp_dir(tmp_regression_images_dir)

        pred_files = glob.glob(f"{output_folder}/*_pred.tif")
        if not pred_files:
            raise FileNotFoundError(
            f"{pred_files} : No prediction file(s) found. Inference failed."
        )

        print(f"Prediction Files: {pred_files}")
        model_output_image = glob.glob(f"{output_folder}/*_pred.tif")[0]

        ######################################################################################################
        ###  (optional) if you want to pass on information to later stages of the pipelines,
        ###             add information to the task config file which will be read later
        ######################################################################################################

        with open(task_config_path, "r") as fp:
            task_dict = json.load(fp)
        task_dict["model_output_image"] = model_output_image

        with open(task_config_path, "w") as fp:
            json.dump(task_dict, fp, indent=4)

    except Exception as ex:
        logger.error(
            f"{inference_id}: Exception {type(ex).__name__}: {ex}",
            stack_info=True,
            exc_info=True,
        )
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1042",  # place holder for error code
            message=f"Inference failed with: {ex}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,
            raise_exception=False,
        )
        raise ex
    finally:
        logger.info(f"{inference_id}: *********Inference execution complete **********")


if __name__ == "__main__":
    run_terratorch_inference()
