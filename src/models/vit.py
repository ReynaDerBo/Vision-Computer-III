import tensorflow as tf
import keras_hub


def build_vit(
    input_shape=(224, 224, 1),
    num_classes=3,
    trainable_layers=0,
    pretrained="vit_base_patch16_224_imagenet",
):

    # ============================================================
    # PRETRAINED / PRESET
    # ============================================================

    if pretrained in [
        None,
        "imagenet",
        "google/vit-base-patch16-224",
        "vit_base_patch16_224",
    ]:
        preset = "vit_base_patch16_224_imagenet"
    else:
        preset = pretrained

    print(f"ViT preset: {preset}")

    # ============================================================
    # INPUT
    # ============================================================

    img_input = tf.keras.layers.Input(
        shape=input_shape,
        name="imagen"
    )

    # ============================================================
    # GRAYSCALE -> RGB
    # ============================================================

    x = tf.keras.layers.Lambda(
        lambda img: tf.image.grayscale_to_rgb(img),
        name="grayscale_to_rgb"
    )(img_input)

    # ============================================================
    # VIT PREENTRENADO
    # ============================================================

    backbone = keras_hub.models.ViTBackbone.from_preset(
        preset
    )

    # ============================================================
    # FINE-TUNING
    # ============================================================

    if trainable_layers is None or trainable_layers == 0:

        backbone.trainable = False

        print("ViT backbone: congelado")

    else:

        backbone.trainable = True

        for layer in backbone.layers:
            layer.trainable = False

        for layer in backbone.layers[-trainable_layers:]:
            layer.trainable = True

        print(
            f"ViT backbone: fine-tuning de las últimas "
            f"{trainable_layers} capas"
        )

    # ============================================================
    # FORWARD PASS
    # ============================================================

    outputs = backbone(x)

    # ============================================================
    # CLS TOKEN
    # ============================================================

    x = outputs[:, 0, :]

    # ============================================================
    # CLASSIFICATION HEAD
    # ============================================================

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="dense_128"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="dense_64"
    )(x)

    output = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classification"
    )(x)

    # ============================================================
    # MODELO FINAL
    # ============================================================

    model = tf.keras.Model(
        inputs=img_input,
        outputs=output,
        name="SmartEye_ViT"
    )

    return model