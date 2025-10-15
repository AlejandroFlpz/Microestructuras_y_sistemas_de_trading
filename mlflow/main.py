import tensorflow as tf
import mlflow

mlflow.set_experiment("CNN Tuning")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Training data shape:", x_train.shape)

def build_model(params):
    model= tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(32, 32, 3)))

    num_filters = params.get('conv_filters', 32)
    conv_layers = params.get('conv_layers', 2)
    activation = params.get('activation', 'relu')

    for _ in range(conv_layers):
        model.add(tf.keras.layers.Conv2D(num_filters, (3, 3), activation=activation))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        num_filters *= 2

    dense_units = params.get('dense_units', 64)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation=activation))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimizer = params.get('optimizer', 'adam')
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

params_space = [
    {'conv_layers': 2, 'conv_filters': 32, 'dense_units': 64, 'activation': 'relu'},
    {'conv_layers': 3, 'conv_filters': 32, 'dense_units': 32, 'activation': 'relu'},
    {'conv_layers': 2, 'conv_filters': 32, 'dense_units': 64, 'activation': 'sigmoid'},
]

print("Training models...")
for params in params_space:
    with mlflow.start_run() as run:
        run_name = f"conv{params['conv_layers']}_filters{params['conv_filters']}_dense{params['dense_units']}_activation{params['activation']}"
        mlflow.log_params(params)

        model = build_model(params)
        hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=2)

        final_metrics = {
            'val_accuracy': hist.history['val_accuracy'][-1],
            'val_loss': hist.history['val_loss'][-1]
        }

        mlflow.log_metrics(final_metrics)
        print(f"Final metrics: {final_metrics}")

        mlflow.tensorflow.log_model(model, "model")