# %%
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

# Load the ONNX model
model = onnx.load("model.onnx")

# Check the model
onnx.checker.check_model(model)

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

# load train_stats from file
train_stats = pd.read_csv("train_stats.csv")
train_stats.columns = train_stats.columns.str.replace("Unnamed: 0", " ")
train_stats = train_stats.set_index(" ")

labels = dataset.pop("MPG")


# Normalize the input data
def norm(x):
    return (x - train_stats["mean"]) / train_stats["std"]


normed_data = norm(dataset)

# export normed_data to csv
normed_data.to_csv("normed_data.csv")

sess = ort.InferenceSession("model.onnx")

# Run the model on the input data
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
input_data = normed_data.to_numpy()
input_data = input_data.astype(np.float32)
predictions = sess.run([label_name], {input_name: input_data})[0]

plt.scatter(labels, predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# %%
