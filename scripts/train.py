from src.models.resnet50 import build_resnet50
from src.models.vit import build_vit

def build_model(config):

    model_name = config["model"]["name"]

    if model_name == "resnet50":

        trainable_layers = (
            config["model"]["trainable_layers"]
            if config["model"]["fine_tuning"]
            else None
        )

        return build_resnet50(
            input_shape=(
                config["data"]["image_size"][0],
                config["data"]["image_size"][1],
                1,
            ),
            num_classes=config["project"]["num_classes"],
            trainable_layers=trainable_layers,
        )

    elif model_name == "vit":

        trainable_layers = (
            config["model"]["trainable_layers"]
            if config["model"]["fine_tuning"]
            else None
        )

        return build_vit(
            input_shape=(
                config["data"]["image_size"][0],
                config["data"]["image_size"][1],
                1,
            ),
            num_classes=config["project"]["num_classes"],
            trainable_layers=trainable_layers,
            pretrained=config["model"]["pretrained"],
        )

    else:
        raise ValueError(
            f"Modelo no soportado: {model_name}"
        )