# Please read me!!!
### A project for Language Processing 2(Final Project) - JianWu(xcb479)
***Note***

```python
ipython notebook demo_BOW.ipynb
```
or 
```python
ipython notebook demo_word2vec.ipynb
```
***The quickly link for Result***

* [Final Result for BOW](https://github.com/JaggerWu/Sentiment-Analysis/blob/master/demo_BOW.ipynb).
* [Final Result for word2vec](https://github.com/JaggerWu/Sentiment-Analysis/blob/master/demo_word2vec.ipynb)

#### About `./sentiment`

I write all available funtion in `./sentiment`

#### Credits

- [fastText](https://github.com/facebookresearch/fastText) by facebook
- the [keras tutorial](https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py) on the subject
- [word2vec](https://radimrehurek.com/gensim/models/word2vec.html)

#### Example output of current model


*A cinematic achievement of amazing depth. Marvellous acting, captivating score,
a journey in a time when directors produced masterpieces. one of a kind, my favorite film,
I LOVE IT!*

gives 0.75

*I went with my girlfriend to the cinema. She's into american comedies and I
really like her, so I just saw the worst most terrible movie in my life.
boring, untalented acting, dull and uninspired cast, absolutely terrible score.
the worst movie ever.*

gives 0.14

*with Robert De Niro and Al Pacino*

gives 0.45

*with Adam Sandler and Jennifer Aniston*

gives  0.38


#### Training

The [data set background](http://lab.homunculus.dk/cgi-bin/LangProc2/news.cgi) for training and testing.

***Also***

You can type
```python
from sentiment.utilities import download_data

download_data()
```
to download dataset

The model for training:
* [SVM](https://en.wikipedia.org/wiki/Support_vector_machine)
* [Random Forest](https://en.wikipedia.org/wiki/Random_forest)


#### Final

I gave up the LSTM since the absolute error was very big As the comments mentioned.  
* [Final Result for BOW](https://github.com/JaggerWu/Sentiment-Analysis/blob/master/demo_BOW.ipynb).
* [Final Result for word2vec](https://github.com/JaggerWu/Sentiment-Analysis/blob/master/demo_word2vec.ipynb)

