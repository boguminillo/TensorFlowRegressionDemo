# %%
# !pip install -q tensorflow==2.15.0 tf2onnx onnxruntime seaborn

# %%
import matplotlib.pyplot as plt
import onnx
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras import layers

# %%
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]
dataset = pd.read_csv(
    "auto-mpg.data",
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)

# %%
dataset.isna().sum()

# %%
dataset = dataset.dropna()

# %%
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

# %%
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# %%
# pop mpg and acceleration into labels
train_labels = train_dataset[["MPG", "Acceleration"]].copy()
train_dataset = train_dataset.drop(columns=["MPG", "Acceleration"])
test_labels = test_dataset[["MPG", "Acceleration"]].copy()
test_dataset = test_dataset.drop(columns=["MPG", "Acceleration"])


# %%
def build_model():
    model = keras.Sequential(
        [
            layers.Normalization(),
            layers.Dense(
                64, activation="relu", input_shape=[len(train_dataset.keys())]
            ),
            layers.Dense(64, activation="relu"),
            layers.Dense(2),
        ]
    )

    model.layers[0].adapt(train_dataset)

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error")
    plt.plot(hist["epoch"], hist["mae"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mae"], label="Val Error")
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.plot(hist["epoch"], hist["mse"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mse"], label="Val Error")
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


# %%
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

EPOCHS = 1000

history = model.fit(
    train_dataset,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, PrintDot()],
)

plot_history(history)

# %%
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))

# %%
test_predictions = model.predict(test_dataset)

# %%
plt.scatter(test_labels["MPG"], test_predictions[:, 0])
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# %%
plt.scatter(test_labels["Acceleration"], test_predictions[:, 1])
plt.xlabel("True Values [Acceleration]")
plt.ylabel("Predictions [Acceleration]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# %%
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

# %%
# export model with h5 format
model.save("model2withNorm.h5")

# %%
# export the model with onnx format
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save_model(onnx_model, "model2withNorm.onnx")
