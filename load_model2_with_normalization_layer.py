# %%
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from tensorflow import keras

# Load the input data
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

dataset = dataset.dropna()
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

labels = dataset[["MPG", "Acceleration"]].copy()
dataset = dataset.drop(columns=["MPG", "Acceleration"])

# %%
# Load the ONNX model
sess = ort.InferenceSession("model2withNorm.onnx")

# Run the model on the input data
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
input_data = dataset.to_numpy()
input_data = input_data.astype(np.double)
predictions = sess.run([label_name], {input_name: input_data})[0]

# %%
# Load the h5 model
model = keras.models.load_model("model2withNorm.h5")

# Run the model on the input data
predictions = model.predict(dataset)

# %%
plt.scatter(labels["MPG"], predictions[:, 0])
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# %%
plt.scatter(labels["Acceleration"], predictions[:, 1])
plt.xlabel("True Values [Acceleration]")
plt.ylabel("Predictions [Acceleration]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# %%
