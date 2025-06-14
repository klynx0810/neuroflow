import json
import h5py
import numpy as np
from .saving_lib import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.model import Model


def save_model_to_h5(model: "Model", filepath: str):
    """
    Lưu toàn bộ model (cấu trúc + trọng số) vào file .h5
    """
    with h5py.File(filepath, "w") as f:
        # 1. Lưu config dưới dạng JSON
        config = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "get_config"):
                layer_config = layer.get_config()
                layer_config["class"] = layer.__class__.__name__
                layer_config["index"] = i
                config.append(layer_config)
        f.attrs["model_config"] = json.dumps(config)

        # 2. Lưu trọng số
        # for i, layer in enumerate(model.layers):
        #     if hasattr(layer, "params"):
        #         group = f.create_group(f"layer_{i}")
        #         for key, val in layer.params.items():
        #             group.create_dataset(key, data=val)
        # 2. Lưu trọng số an toàn
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "params") and layer.params:
                group_name = f"{i:02d}_{layer.name}"
                # group = f.create_group(f"layer_{i}_{layer.name}")
                group = f.create_group(group_name)
                for key, val in layer.params.items():
                    if val is not None:
                        group.create_dataset(key, data=val.astype(val.dtype))
                        print(f"Saved {key} of {group_name} shape={val.shape}")


def load_model_from_h5(filepath: str, model_class: "Model"):
    """
    Tải model từ file .h5 đã lưu.
    Cần truyền vào class gốc, ví dụ: Sequential
    """
    with h5py.File(filepath, "r") as f:
        config = json.loads(f.attrs["model_config"])
        model: "Model" = model_class()

        for layer_cfg in config:
            layer_class = layer_cfg.pop("class")
            layer_cfg.pop("index", None)
            if 'input_shape' in layer_cfg and layer_cfg['input_shape'] is not None:
                layer_cfg['input_shape'] = tuple(layer_cfg['input_shape'])

            if layer_class == "Conv2D":
                model.add(Conv2D(**layer_cfg))
            elif layer_class == "Dense":
                model.add(Dense(**layer_cfg))
            elif layer_class == "Flatten":
                model.add(Flatten(**layer_cfg))
            elif layer_class == "Dropout":
                model.add(Dropout(**layer_cfg))
            elif layer_class == "MaxPooling2D":
                model.add(MaxPooling2D(**layer_cfg))
            elif layer_class == "Activation":
                # print("Activation layer config:", layer_cfg)
                model.add(Activation(**layer_cfg))
            else:
                raise ValueError(f"Unknown layer class: {layer_class}")

        # for i, layer in enumerate(model.layers):
        #     print(layer.params)
        # Load trọng số
        # for i, layer in enumerate(model.layers):
        #     if hasattr(layer, "params"):
        #         for key in layer.params:
        #             layer.params[key] = f[f"layer_{i}"][key][...]

        # for gr in f:
        #     print(gr)
        # 2. Load trọng số vào từng layer
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "params"):
                group_name = f"{i:02d}_{layer.name}"
                if group_name in f:
                    layer_group = f[group_name]
                    layer.params = {}  # reset để load mới toàn bộ
                    params = {}
                    for key in layer_group:
                        data = layer_group[key][()]
                        params[key] = data
                    layer.set_params(params)
                    layer.built = True
                    print(f"Loaded W for {layer.name}:", params["W"].ravel()[:5])
                    print(f"Current W in layer:", layer.params["W"].ravel()[:5])
                    print(f"Set params for {group_name}: keys = {list(params.keys())}")
                else:
                    print(f"Group '{group_name}' không tồn tại trong file")


        # for i, layer in enumerate(model.layers):
        #     print(layer.params)

    return model

def show_model(filepath: str, model_config=False):
    with h5py.File(filepath, "r") as f:
        if model_config:
            config = json.loads(f.attrs["model_config"])
            print("Model config:")
            for layer_cfg in config:
                print(layer_cfg)
        else:
            print("Các group layer trong file:")
            for layer_name in sorted(f.keys()):
                if layer_name[0:2].isdigit():
                    print(f"\n📦 {layer_name}")
                    layer_group = f[layer_name]
                    for param_name in layer_group:
                        param_value = layer_group[param_name][()]
                        print(f"  - {param_name}: shape {param_value.shape}")