from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET = "rating"
NUMERIC_FEATURES = ["release_year"]
CATEGORICAL_FEATURES = [
	"type",
	"country",
	"director",
	"duration",
	"listed_in",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
	"""Load content records and remove rows without a rating label."""
	dataset_path = Path(__file__).with_name("Dataset.csv")
	df = pd.read_csv(dataset_path)
	df = df.dropna(subset=[TARGET]).copy()
	df[TARGET] = df[TARGET].astype(str).str.strip()
	return df[df[TARGET] != ""]


def build_preprocessor() -> ColumnTransformer:
	return ColumnTransformer(
		transformers=[
			(
				"numeric",
				Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
				NUMERIC_FEATURES,
			),
			(
				"categorical",
				Pipeline(
					steps=[
						(
							"imputer",
							SimpleImputer(strategy="most_frequent"),
						),
						(
							"encoder",
							OneHotEncoder(
								handle_unknown="ignore",
								sparse_output=True,
							),
						),
					]
				),
				CATEGORICAL_FEATURES,
			),
		]
	)


def make_pipeline(classifier) -> Pipeline:
	return Pipeline(
		steps=[
			("preprocessor", build_preprocessor()),
			("classifier", classifier),
		]
	)


def evaluate_model(name: str, model: Pipeline, X_test, y_test) -> dict:
	predictions = model.predict(X_test)
	accuracy = accuracy_score(y_test, predictions)
	weighted_f1 = f1_score(y_test, predictions, average="weighted")

	print(f"\n{name}")
	print(f"Accuracy: {accuracy:.3f}")
	print(f"Weighted F1: {weighted_f1:.3f}")
	print(classification_report(y_test, predictions, zero_division=0))

	return {
		"Model": name,
		"Accuracy": accuracy,
		"Weighted F1": weighted_f1,
	}


def main() -> None:
	df = load_data()
	print("Rating category distribution:")
	print(df[TARGET].value_counts().to_string())

	X = df[FEATURES]
	y = df[TARGET]
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=y,
	)

	models = {
		"Decision Tree": make_pipeline(
			DecisionTreeClassifier(
				max_depth=20,
				min_samples_leaf=2,
				random_state=RANDOM_STATE,
			)
		),
		"Random Forest": make_pipeline(
			RandomForestClassifier(
				n_estimators=150,
				min_samples_leaf=2,
				random_state=RANDOM_STATE,
				n_jobs=-1,
			)
		),
	}

	results = []
	for name, model in models.items():
		model.fit(X_train, y_train)
		results.append(evaluate_model(name, model, X_test, y_test))

	tuned_model = make_pipeline(
		RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
	)
	search = GridSearchCV(
		tuned_model,
		param_grid={
			"classifier__n_estimators": [100, 150],
			"classifier__max_depth": [None, 20],
			"classifier__min_samples_leaf": [1, 2],
		},
		cv=3,
		scoring="accuracy",
		n_jobs=-1,
		verbose=0,
	)
	search.fit(X_train, y_train)
	print("\nBest Random Forest parameters:")
	print(search.best_params_)
	print(f"Best cross-validation accuracy: {search.best_score_:.3f}")
	results.append(
		evaluate_model("Tuned Random Forest", search.best_estimator_, X_test, y_test)
	)

	comparison = pd.DataFrame(results).sort_values(
		by="Accuracy", ascending=False
	)
	print("\nModel accuracy comparison:")
	print(comparison.to_string(index=False))

	best_predictions = search.best_estimator_.predict(X_test)
	print("\nTuned model confusion matrix labels:")
	print(sorted(y_test.unique()))
	print(confusion_matrix(y_test, best_predictions))


if __name__ == "__main__":
	main()
