# From https://github.com/IBM/terratorch/blob/main/examples/scripts/fix_backwards_compatibility.py
import torch
import structlog

logger = structlog.get_logger()

def convert_old_checkpoints(checkpoint_path: str):

    converted_checkpoint_path = (checkpoint_path).split('.')[0]+'_Fixed.'+(checkpoint_path).split('.')[1]
    logger.info(f"Path In : {checkpoint_path},  Path Out: {converted_checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict_renamed = {}


    for k, v in state_dict.items():
        # remove the module. part
        if k == 'state_dict':
            state_dict_renamed[k] = {}
            for k1, v1 in v.items():
                splits = k1.split(".")
                splits_ = [s for s in splits if "timm" not in s]
                k1_ = ".".join(splits_)
                if k1 != k1_:
                    state_dict_renamed[k][k1_] = v1
                else:
                    state_dict_renamed[k][k1] = v1
        else:
            state_dict_renamed[k] = v

    torch.save(state_dict_renamed, converted_checkpoint_path)

    return converted_checkpoint_path

