from sentiment.utilities import interactive_session
from sentiment.utilities import predict_from_texts


example_sentences = [
    'A cinematic achievement of amazing depth. Marvellous acting, captivating '
    'score, a journey in a time when directors produced masterpieces. one of '
    'a kind, my favorite film, I LOVE IT!',
    'I went with my girlfriend to the cinema. She\'s into american comedies '
    'and I really like her, so I just saw the worst most terrible movie in '
    'my life. boring, untalented acting, dull and uninspired cast, '
    'absolutely terrible score. the worst movie ever.',
    'with Robert De Niro and Al Pacino',
    'with Adam Sandler and Jennifer Aniston'
]

predictions = predict_from_texts(example_sentences)

print("Some examples:")
for text, prediction in zip(example_sentences, predictions):
    print("For input: %s" % text)
    print("    ---> %.2f" % prediction)

interactive_session()
