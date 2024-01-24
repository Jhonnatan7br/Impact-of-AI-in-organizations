# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_analyze_sentiment.py

DESCRIPTION:
    This sample demonstrates how to analyze sentiment in documents.
    An overall and per-sentence sentiment is returned.

    In this sample we will be a skydiving company going through reviews people have left for our company.
    We will extract the reviews that we are certain have a positive sentiment and post them onto our
    website to attract more divers.

USAGE:
    python sample_analyze_sentiment.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_LANGUAGE_ENDPOINT - the endpoint to your Language resource.
    2) AZURE_LANGUAGE_KEY - your Language subscription key
"""
import pandas as pd
research = pd.read_csv("C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/Datasets/scopus.csv")
research.head()
research.describe()
# Create a sub-dataset with the first 10 lines
sub_dataset = research.head(10)


def sample_analyze_sentiment() -> None:
    print(
        "In this sample we will be combing through reviews customers have left about their"
        "experience using our skydiving company, Contoso."
    )
    print(
        "We start out with a list of reviews. Let us extract the reviews we are sure are "
        "positive, so we can display them on our website and get even more customers!"
    )

    # [START analyze_sentiment]
    import os
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.textanalytics import TextAnalyticsClient
    from dotenv import load_dotenv


    # Load environment variables from the .env file
    load_dotenv()
    # Access the environment variables
    endpoint = os.getenv("AZURE_TEXT_ANALYSIS_ENDPOINT")
    key = os.getenv("AZURE_TEXT_ANALYSIS_KEY")
    
    #endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    #key = os.environ["AZURE_LANGUAGE_KEY"]

    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    # Example
    documents = [
        """I had the best day of my life. I decided to go sky-diving and it made me appreciate my whole life so much more.
        I developed a deep-connection with my instructor as well, and I feel as if I've made a life-long friend in her.""",
        """This was a waste of my time. All of the views on this drop are extremely boring, all I saw was grass. 0/10 would
        not recommend to any divers, even first timers.""",
        """This was pretty good! The sights were ok, and I had fun with my instructors! Can't complain too much about my experience""",
        """I only have one word for my experience: WOW!!! I can't believe I have had such a wonderful skydiving company right
        in my backyard this whole time! I will definitely be a repeat customer, and I want to take my grandmother skydiving too,
        I know she'll love it!"""
    ]
    
    # Extract descriptions from the 'description' column of the dataframe
    documents = sub_dataset['Abstract'].tolist()
    documents = [f'"""{doc}"""' for doc in documents]

    # Extract descriptions from the 'Abstract' column of the dataframe


    result = text_analytics_client.analyze_sentiment(documents, show_opinion_mining=True)
    docs = [doc for doc in result if not doc.is_error]

    print("Let's visualize the sentiment of each of these documents")
    for idx, doc in enumerate(docs):
        print(f"Document text: {documents[idx]}")
        print(f"Overall sentiment: {doc.sentiment}")
    # [END analyze_sentiment]

    print("Now, let us extract all of the positive reviews")
    positive_reviews = [doc for doc in docs if doc.sentiment == 'positive']

    print("We want to be very confident that our reviews are positive since we'll be posting them on our website.")
    print("We're going to confirm our chosen reviews are positive using two different tests")

    print(
        "First, we are going to check how confident the sentiment analysis model is that a document is positive. "
        "Let's go with a 90% confidence."
    )
    positive_reviews = [
        review for review in positive_reviews
        if review.confidence_scores.positive >= 0.9
    ]

    print(
        "Finally, we also want to make sure every sentence is positive so we only showcase our best selves!"
    )
    positive_reviews_final = []
    for idx, review in enumerate(positive_reviews):
        print(f"Looking at positive review #{idx + 1}")
        any_sentence_not_positive = False
        for sentence in review.sentences:
            print("...Sentence '{}' has sentiment '{}' with confidence scores '{}'".format(
                sentence.text,
                sentence.sentiment,
                sentence.confidence_scores
                )
            )
            if sentence.sentiment != 'positive':
                any_sentence_not_positive = True
        if not any_sentence_not_positive:
            positive_reviews_final.append(review)

    print("We now have the final list of positive reviews we are going to display on our website!")


if __name__ == '__main__':
    sample_analyze_sentiment()