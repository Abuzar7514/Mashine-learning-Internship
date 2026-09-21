from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "high_reach_proxy"
BROAD_AUDIENCE_RATINGS = {"TV-14", "TV-MA", "PG-13", "R"}
NUMERIC_FEATURES = [
	"release_year",
	"duration_value",
	"genre_count",
	"country_count",
	"title_length",
	"release_age",
	"date_added_year",
	"date_added_month",
	"is_movie",
	"has_director",
]
CATEGORICAL_FEATURES = ["type", "country", "listed_in"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_engineer_features() -> pd.DataFrame:
	"""Create modeling features and an explicit engagement-free success proxy."""
	dataset_path = Path(__file__).with_name("Dataset.csv")
	df = pd.read_csv(dataset_path, parse_dates=["date_added"])
	df = df.dropna(subset=["rating"]).copy()

	df["duration_value"] = pd.to_numeric(
		df["duration"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
	)
	df["genre_count"] = df["listed_in"].fillna("").str.count(",") + 1
	df["country_count"] = df["country"].fillna("").str.count(",") + 1
	df["title_length"] = df["title"].fillna("").str.len()
	df["has_director"] = df["director"].notna().astype(int)
	df["is_movie"] = (df["type"] == "Movie").astype(int)
	df["date_added_year"] = df["date_added"].dt.year
	df["date_added_month"] = df["date_added"].dt.month
	df["release_age"] = df["date_added"].dt.year - df["release_year"]
	df[TARGET] = df["rating"].isin(BROAD_AUDIENCE_RATINGS).astype(int)
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
							OneHotEncoder(handle_unknown="ignore", sparse_output=False),
						),
					]
				),
				CATEGORICAL_FEATURES,
			),
		]
	)


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
	X = df[FEATURES]
	y = df[TARGET]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
	)
	models = {
		"Logistic Regression": LogisticRegression(
			max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
		),
		"Random Forest": RandomForestClassifier(
			n_estimators=200,
			min_samples_leaf=2,
			class_weight="balanced",
			random_state=RANDOM_STATE,
			n_jobs=-1,
		),
	}
	results = []
	trained_models = {}
	for name, classifier in models.items():
		pipeline = Pipeline(
			steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
		)
		pipeline.fit(X_train, y_train)
		predictions = pipeline.predict(X_test)
		results.append(
			{
				"Model": name,
				"Accuracy": accuracy_score(y_test, predictions),
				"Precision": precision_score(y_test, predictions, zero_division=0),
				"Recall": recall_score(y_test, predictions, zero_division=0),
				"F1": f1_score(y_test, predictions, zero_division=0),
			}
		)
		trained_models[name] = pipeline

	comparison = pd.DataFrame(results).sort_values("F1", ascending=False)
	return comparison, trained_models, pd.DataFrame({"actual": y_test, "predicted": predictions})


def generate_insights(df: pd.DataFrame, model: Pipeline) -> list[str]:
	proxy_rate = df[TARGET].mean()
	type_summary = df.groupby("type")[TARGET].agg(["mean", "count"]).sort_values("mean")
	top_type = type_summary["mean"].idxmax()
	country_summary = (
		df.assign(country=df["country"].fillna("Unknown").str.split(", "))
		.explode("country")
		.groupby("country")[TARGET]
		.agg(["mean", "count"])
		.query("count >= 50")
		.sort_values("mean", ascending=False)
	)
	top_country = country_summary.index[0] if not country_summary.empty else "Unavailable"
	year_summary = df.groupby("release_year")[TARGET].mean()
	latest_year = int(year_summary.index.max())
	latest_rate = year_summary.loc[latest_year]

	preprocessor = model.named_steps["preprocessor"]
	classifier = model.named_steps["classifier"]
	feature_names = preprocessor.get_feature_names_out()
	if hasattr(classifier, "feature_importances_"):
		importance_values = classifier.feature_importances_
	else:
		importance_values = np.abs(classifier.coef_[0])
	importance = pd.Series(importance_values, index=feature_names)
	top_features = ", ".join(
		importance.sort_values(ascending=False).head(5).index.str.replace(
			"numeric__", "", regex=False
		).str.replace("categorical__", "", regex=False)
	)
	return [
		f"The high-reach proxy applies to {proxy_rate:.1%} of catalog titles.",
		f"{top_type} content has the highest proxy rate at {type_summary.loc[top_type, 'mean']:.1%}.",
		f"Among countries with at least 50 titles, {top_country} has the highest proxy rate.",
		f"Titles released in {latest_year} have a {latest_rate:.1%} proxy rate.",
		f"The strongest model signals are: {top_features}.",
	]


def save_dashboard(
	df: pd.DataFrame, comparison: pd.DataFrame, output_path: Path
) -> None:
	fig, axes = plt.subplots(2, 2, figsize=(13, 9))
	comparison.set_index("Model")[["Accuracy", "F1"]].plot.bar(ax=axes[0, 0])
	axes[0, 0].set_title("Model performance")
	axes[0, 0].set_ylim(0, 1)
	axes[0, 0].set_ylabel("Score")

	type_rates = df.groupby("type")[TARGET].mean().sort_values()
	type_rates.plot.barh(ax=axes[0, 1], color="#3b82f6")
	axes[0, 1].set_title("Proxy rate by content type")
	axes[0, 1].set_xlabel("Proxy rate")

	year_rates = df.groupby("release_year")[TARGET].mean()
	year_rates.plot(ax=axes[1, 0], color="#e76f51")
	axes[1, 0].set_title("Proxy rate by release year")
	axes[1, 0].set_ylabel("Proxy rate")

	genre_rates = (
		df.assign(genre=df["listed_in"].fillna("").str.split(", "))
		.explode("genre")
		.groupby("genre")[TARGET]
		.agg(["mean", "count"])
		.query("count >= 50")
		.sort_values("mean", ascending=False)
		.head(8)
		.sort_values("mean")
	)
	genre_rates["mean"].plot.barh(ax=axes[1, 1], color="#2a9d8f")
	axes[1, 1].set_title("Top genre proxy rates")
	axes[1, 1].set_xlabel("Proxy rate")
	fig.suptitle("Netflix Content Success Analytics Dashboard", fontsize=16)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def main() -> None:
	df = load_and_engineer_features()
	print("Success proxy: rating in TV-14, TV-MA, PG-13, or R")
	print(f"Analyzing {len(df)} titles with {len(FEATURES)} engineered feature groups.")
	comparison, models, _ = evaluate_models(df)
	print("\nModel comparison:")
	print(comparison.to_string(index=False, float_format="%.3f"))

	best_model_name = comparison.iloc[0]["Model"]
	insights = generate_insights(df, models[best_model_name])
	print("\nAutomated business insights:")
	for insight in insights:
		print(f"- {insight}")

	output_dir = Path(__file__).parent
	comparison.to_csv(output_dir / "task_6_model_comparison.csv", index=False)
	(output_dir / "task_6_business_insights.txt").write_text(
		"Netflix Content Success Analytics\n\n" + "\n".join(insights) + "\n",
		encoding="utf-8",
	)
	save_dashboard(df, comparison, output_dir / "task_6_success_dashboard.png")
	print("\nReports saved: task_6_model_comparison.csv, task_6_business_insights.txt")
	print("Dashboard saved: task_6_success_dashboard.png")


if __name__ == "__main__":
	main()
