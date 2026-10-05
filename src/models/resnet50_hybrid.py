import tensorflow as tf

def build_resnet50_hybrid(
    input_shape=(224, 224, 1),
    num_classes=3,
    trainable_layers=None,
):
    # Image branch
    img_input = tf.keras.layers.Input(
        shape=input_shape, name="imagen"
    )

    x = tf.keras.layers.Lambda(
        lambda img: tf.image.grayscale_to_rgb(img)
    )(img_input)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False
    if trainable_layers:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Metadata branches
    tipo_a_input = tf.keras.layers.Input(
        shape=(1,), dtype=tf.float32, name="tipo A"
    )
    tipo_b_input = tf.keras.layers.Input(
        shape=(1,), dtype=tf.float32, name="tipo B"
    )

    t_a = tf.keras.layers.Dense(64, activation="relu")(tipo_a_input)
    t_b = tf.keras.layers.Dense(64, activation="relu")(tipo_b_input)

    combined = tf.keras.layers.Concatenate()([x, t_a, t_b])
    combined = tf.keras.layers.Dense(128, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)
    output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classification"
    )(combined)

    return tf.keras.Model(
        inputs=[img_input, tipo_a_input, tipo_b_input],
        outputs=output,
        name="SmartEye_ResNet50_Hybrid",
    )
