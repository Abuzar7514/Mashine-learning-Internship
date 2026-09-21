from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
CLUSTER_RANGE = range(2, 9)
SILHOUETTE_SAMPLE_SIZE = 2000
NUMERIC_FEATURES = ["release_year", "duration_value"]
CATEGORICAL_FEATURES = ["type", "rating", "country", "listed_in"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
	"""Load the Netflix catalog and create a numeric duration feature."""
	dataset_path = Path(__file__).with_name("Dataset.csv")
	df = pd.read_csv(dataset_path)
	df["duration_value"] = pd.to_numeric(
		df["duration"].astype(str).str.extract(r"(\d+)")[0],
		errors="coerce",
	)
	return df


def build_preprocessor() -> ColumnTransformer:
	return ColumnTransformer(
		transformers=[
			(
				"numeric",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="median")),
						("scaler", StandardScaler()),
					]
				),
				NUMERIC_FEATURES,
			),
			(
				"categorical",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						(
							"encoder",
							OneHotEncoder(
								handle_unknown="ignore",
								sparse_output=False,
							),
						),
					]
				),
				CATEGORICAL_FEATURES,
			),
		]
	)


def select_cluster_count(features) -> tuple[int, dict[int, float]]:
	silhouette_scores = {}
	for cluster_count in CLUSTER_RANGE:
		print(f"Evaluating {cluster_count} clusters...", flush=True)
		model = KMeans(
			n_clusters=cluster_count,
			random_state=RANDOM_STATE,
			n_init=10,
		)
		labels = model.fit_predict(features)
		silhouette_scores[cluster_count] = silhouette_score(
			features,
			labels,
			sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(features)),
			random_state=RANDOM_STATE,
		)

	best_count = max(silhouette_scores, key=silhouette_scores.get)
	return best_count, silhouette_scores


def print_cluster_profiles(df: pd.DataFrame, labels) -> None:
	profile = df.assign(cluster=labels).groupby("cluster").agg(
		Titles=("show_id", "count"),
		Average_Release_Year=("release_year", "mean"),
		Average_Duration=("duration_value", "mean"),
		Movie_Share=("type", lambda values: (values == "Movie").mean()),
		Most_Common_Rating=("rating", lambda values: values.mode().iat[0]),
	)
	profile["Movie_Share"] = profile["Movie_Share"].map(lambda value: f"{value:.1%}")
	print("\nCluster profiles:")
	print(profile.round({"Average_Release_Year": 1, "Average_Duration": 1}).to_string())


def save_cluster_visualization(features, labels, output_path: Path) -> None:
	reduced_features = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(features)
	plt.figure(figsize=(10, 6))
	plot = plt.scatter(
		reduced_features[:, 0],
		reduced_features[:, 1],
		c=labels,
		cmap="viridis",
		alpha=0.65,
		s=18,
	)
	plt.colorbar(plot, label="Cluster")
	plt.title("Netflix Content Segments")
	plt.xlabel("Principal Component 1")
	plt.ylabel("Principal Component 2")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def main() -> None:
	df = load_data()
	print(f"Loaded {len(df)} Netflix titles.", flush=True)
	preprocessor = build_preprocessor()
	features = preprocessor.fit_transform(df[FEATURES])
	print(f"Prepared feature matrix with shape {features.shape}.", flush=True)

	best_count, silhouette_scores = select_cluster_count(features)
	print("Silhouette scores by cluster count:")
	for cluster_count, score in silhouette_scores.items():
		print(f"{cluster_count} clusters: {score:.3f}")
	print(f"\nSelected cluster count: {best_count}")

	model = KMeans(
		n_clusters=best_count,
		random_state=RANDOM_STATE,
		n_init=10,
	)
	labels = model.fit_predict(features)
	print_cluster_profiles(df, labels)

	visualization_path = Path(__file__).with_name("netflix_content_clusters.png")
	save_cluster_visualization(features, labels, visualization_path)
	print(f"\nCluster visualization saved to: {visualization_path.name}")


if __name__ == "__main__":
	main()
