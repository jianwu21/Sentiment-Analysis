""" Sentiment analysis framework, I like writing titles """
from .utilities import (
    download_data,
    load_data,
    generate_word2vec,
    load_word2vec,
    text2vec,
    generate_dataset
    )

from .model import make_model, training_session
