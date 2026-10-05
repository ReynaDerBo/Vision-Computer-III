import tensorflow as tf

def image_generators(dataset_dir, batch_size=2, image_size=(224, 224),
                     seed=42, use_clahe=False, preprocessing_function=None):

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        preprocessing_function=preprocessing_function if use_clahe else None
    )

    train_dir = f"{dataset_dir}/train"
    val_dir = f"{dataset_dir}/val"
    test_dir = f"{dataset_dir}/test"

    common = dict(
        target_size=image_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        seed=seed,
    )

    train = generator.flow_from_directory(
        train_dir, shuffle=True, **common
    )
    val = generator.flow_from_directory(
        val_dir, shuffle=False, **common
    )
    test = generator.flow_from_directory(
        test_dir, shuffle=False, **common
    )

    return train, val, test


# Placeholder for the multimodal generator from the original notebook.
# Keep the original implementation as the reference while adapting it here.
def generator_with_features_2(*args, **kwargs):
    raise NotImplementedError(
        "Move/adapt the multimodal generator from SmartEye_original.ipynb here."
    )
