import tensorflow as tf

def build_resnet50(
    input_shape=(224, 224, 1),
    num_classes=3,
    trainable_layers=40,
):
    img_input = tf.keras.layers.Input(
        shape=input_shape,
        name="imagen"
    )

    x = tf.keras.layers.Lambda(
        lambda img: tf.image.grayscale_to_rgb(img)
    )(img_input)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    if trainable_layers is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classification"
    )(x)

    return tf.keras.Model(img_input, output, name="SmartEye_ResNet50")
