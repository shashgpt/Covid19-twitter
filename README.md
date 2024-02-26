# Covid19-twitter dataset
We constructed a dataset from tweets based on Covid-19 topic for sentence-level binary sentiment classification task. The dataset can be downloaded from this google-drive link: https://drive.google.com/file/d/1p3yZ-L8OxsZYuXOJoOn8aspYFLnn68b_/view?usp=share_link

Below, we describe each column in the dataset
| Columns        | Meaning      |
| ------------- |:-------------:|
| tweet_id      | The tweet ID of the scrapped tweet |
| rule_structure | The Rule-syntactic structure present in the tweet |
| rule_label | Applicable Rule-label corresponding to the Rule-syntactic structure present |
| contrast | Indicates whether the conjuncts of the preprocessed tweet have contrastive polarities of sentiment incase there is a Rule-label applicable on the tweet |
| clause_A | The A conjunct of the preprocessed tweet if the tweet contains a Rule-label |
| clause_B | The B conjunct of the preprocessed tweet if the tweet contains a Rule-label |
| emojis | All the emojis present in the tweet |
| emoji_names | Description of emojis in the tweet |
| emotion_scores | Correponding negative or positive emotion score values calculated as per EmoTag table described in the paper |
| agg_emotion_score | Arithemic addition of emotion scores |
| sentiment_label | Assigned Sentiment polarity label as per the sentiment labeling methodology described in the paper |
| vader_score_sentence | VADER sentiment analysis tool score for the preprocessed tweet |
| vader_score_clause_A | VADER sentiment analysis tool score for the A conjunct in tweet if the tweet contains a Rule-label |
| vader_score_clause_B | VADER sentiment analysis tool score for the B conjunct in tweet if the tweet contains a Rule-label |
| vader_sentiment_sentence | Sentiment polarity of the preprocessed tweet determined as per the VADER sentiment analysis tool |
| vader_sentiment_clause_A | Sentiment polarity of the A conjunct in the tweet determined as per the VADER sentiment analysis tool |
| vader_sentiment_clause_B | Sentiment polarity of the B conjunct in the tweet determined as per the VADER sentiment analysis tool |
