# Sentiment Analysis
###### A project for Peter's class

### What's new?

###### Function to retrieve the movies dataset

    from sentiment import download_data
    download_data()

###### Read them!

    from sentiment import load_data
    # HMMM actually I should check that there are downloaded data first...
    data = load_data()
    texts = data.text
