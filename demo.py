import numpy as np

from sentiment import (download_data, load_data, text_to_chars,
                       dataset_split, god_model, training_session)


# First download the data
download_data()

# Load them
data = load_data()

# Get X and y values
X = np.array([text_to_chars(x) for x in data.text.values])
y = data.ratings.values

# Initialize the model
model = god_model(X.shape)

# Get train / validation / test set indexes
idc_all = dataset_split(y)

# Train
err, model = training_session(model, X, y, idc_all)
