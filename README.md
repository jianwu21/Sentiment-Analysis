# Sentiment Analysis
###### A project for Peter's class

#### Help on install

    pip install -r requirements.txt

OR

    pip install --user -r requirements.txt

### What's new?

###### Function to retrieve the movies dataset

    from sentiment import download_data
    download_data()

###### Read them!

    from sentiment import load_data
    # HMMM actually I should check that there are downloaded data first...
    data = load_data()
    texts = data.text

###### Convert them to vectors

    from sentiment import generate_word2vec
    model = generate_word2vec(data.text.values, size=100)
    # as well as
    model = load_word2vec()
    print(model['word'])
