import tensorflow as tf

def compile_model(model, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model,
    train_data,
    val_data,
    epochs=30,
    callbacks=None,
):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks or [],
    )
    return history
