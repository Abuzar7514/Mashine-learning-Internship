import pandas as pd

df = pd.read_csv("Dataset.csv")

df = df.fillna("")

print(df.head())

df["content"] = (
    df["listed_in"] + " " +
    df["director"] + " " +
    df["type"]
)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(df["content"])


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(tfidf_matrix)

def get_title_index(title):
    matches = df.index[df["title"].str.casefold() == title.casefold()]
    if len(matches) == 0:
        raise ValueError(f"Title not found: {title}")
    return matches[0]


def recommend_with_scores(title, n=10):
    index = get_title_index(title)
    scores = enumerate(similarity[index])
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    source_title = df.iloc[index]["title"].casefold()
    return [
        (df.iloc[i]["title"], score, i)
        for i, score in scores
        if df.iloc[i]["title"].casefold() != source_title
    ][:n]


def recommend(title, n=10):
    return [title for title, _, _ in recommend_with_scores(title, n)]


def is_relevant(source_index, recommendation_index):
    source_categories = set(df.iloc[source_index]["listed_in"].split(", "))
    recommendation_categories = set(
        df.iloc[recommendation_index]["listed_in"].split(", ")
    )
    return (
        df.iloc[source_index]["type"] == df.iloc[recommendation_index]["type"]
        and bool(source_categories & recommendation_categories)
    )


def precision_at_k(title, k=5):
    source_index = get_title_index(title)
    recommendations = recommend_with_scores(title, k)
    relevant_count = sum(
        is_relevant(source_index, recommendation_index)
        for _, _, recommendation_index in recommendations
    )
    return relevant_count / len(recommendations) if recommendations else 0.0


def average_similarity(title, k=5):
    scores = recommend_with_scores(title, k)
    return sum(score for _, score, _ in scores) / len(scores) if scores else 0.0


title = "Stranger Things"
recommendations = recommend_with_scores(title, 5)

print(f"Recommendations for {title}:")
for recommendation, score, _ in recommendations:
    print(f"- {recommendation} (similarity: {score:.3f})")

print(f"Precision@5: {precision_at_k(title, 5):.3f}")
print(f"Average similarity@5: {average_similarity(title, 5):.3f}")