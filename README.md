# Covid19-twitter dataset
We constructed a dataset from tweets based on the COVID-19 topic for the sentence-level binary sentiment classification task. As per the Twitter Developer Terms (check the Content redistribution section on https://developer.twitter.com/en/developer-terms/agreement-and-policy), the actual tweets cannot be released, however, we are releasing the tweet IDs, which can be used to get actual tweets using a python package called Tweepy (https://www.tweepy.org/). To do so, you will need a Twitter Developer Account with "Basic" level prvileges (https://developer.x.com/en/portal/products/basic), which contains access to the "GET /2/tweets" API-endpoint. The dataset can be downloaded from the Zenodo platform: https://zenodo.org/records/11482356

Below, we describe each column in the dataset
| Columns        | Meaning      |
| ------------- |:-------------:|
| tweet_id      | The tweet ID of the scrapped tweet |
| rule_structure | The Rule-syntactic structure present in the tweet |
| rule_label | Applicable Rule-label corresponding to the Rule-syntactic structure present |
| contrast | Indicates whether the conjuncts of the preprocessed tweet have contrastive polarities of sentiment in case there is a Rule label applicable on the tweet |
| emojis | All the emojis present in the tweet |
| emoji_names | Description of emojis in the tweet |
| emotion_scores | Corresponding negative or positive emotion score values calculated as per the EmoTag table described in the paper |
| agg_emotion_score | Arithemic addition of emotion scores |
| sentiment_label | Assigned Sentiment polarity label as per the sentiment labelling methodology described in the paper |
| vader_score_sentence | VADER sentiment analysis tool score for the preprocessed tweet |
| vader_score_clause_A | VADER sentiment analysis tool score for the A conjunct in tweet if the tweet contains a Rule-label |
| vader_score_clause_B | VADER sentiment analysis tool score for the B conjunct in tweet if the tweet contains a Rule-label |
| vader_sentiment_sentence | Sentiment polarity of the preprocessed tweet determined as per the VADER sentiment analysis tool |
| vader_sentiment_clause_A | Sentiment polarity of the A conjunct in the tweet determined as per the VADER sentiment analysis tool |
| vader_sentiment_clause_B | Sentiment polarity of the B conjunct in the tweet determined as per the VADER sentiment analysis tool |

## Steps to reproduce benchmark results in the paper
1) Ensure that you have created a Twitter developer account at: <https://developer.x.com/>. You will need to upgrade it to a "Basic" tier in order to get access to the "GET /2/tweets" API-endpoint. Given the tweet IDs, this will let you get the actual tweets from Twitter platform (check https://developer.x.com/en/docs/api-reference-index).
2) After creating the Twitter developer account, identify the following information related to your default project: -
   a) Bearer token
   b) API key
   c) API secret
   d) Access token
   e) Access token secret
   <img width="1686" alt="image" src="https://github.com/shashgpt/Covid19-twitter/assets/22150410/be8a93c1-f2a6-4669-bf19-31aca409478f">
3) Install the dependencies listed in requirements.txt file, and run the check_reproducibility.ipynb Jupyter notebook. Please note that the notebook was tested on a RTX2080 ti GPU, Ubuntu 22.04.2 LTS operating system, and is written in Python 3.11.0rc1 language.
4) Alternatively, you can download the Docker image (https://hub.docker.com/r/sunbro/covid19_twitter/tags) instead of maunally creating the runtime environment. Instructions on installing and building a GPU-enabled Docker image can be found here: https://www.tensorflow.org/install/docker#examples_using_cpu-only_images

## Check the existing results in the paper
If you would like to inspect the existing results presented in the paper, please download the file from this link: https://drive.google.com/file/d/1i6zXPoLFT5c_SRkLmRJG76dIENst6ZZ8/view?usp=sharing

unzip it, and run the suitable code block in the notebook

