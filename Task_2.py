import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("Dataset.csv")
df = df.dropna(subset=["type"]).copy()

target = "type"
numeric_features = ["release_year"]
categorical_features = [
	"country",
	"director",
	"rating",
	"duration",
	"listed_in",
]
features = numeric_features + categorical_features

X = df[features]
y = df[target]

numeric_pipeline = Pipeline(
	steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_pipeline = Pipeline(
	steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
	]
)

preprocessor = ColumnTransformer(
	transformers=[
		("numeric", numeric_pipeline, numeric_features),
		("categorical", categorical_pipeline, categorical_features),
	]
)

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y,
)

models = {
	"Logistic Regression": LogisticRegression(
		max_iter=1000,
		random_state=42,
	),
	"Random Forest": RandomForestClassifier(
		n_estimators=100,
		random_state=42,
		n_jobs=-1,
	),
}

results = []

for name, model in models.items():
	pipeline = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("classifier", model),
		]
	)
	pipeline.fit(X_train, y_train)
	predictions = pipeline.predict(X_test)
	accuracy = accuracy_score(y_test, predictions)
	results.append({"Model": name, "Accuracy": accuracy})

	print(f"\n{name}")
	print(f"Accuracy: {accuracy:.3f}")
	print(classification_report(y_test, predictions, zero_division=0))

comparison = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("Model accuracy comparison:")
print(comparison.to_string(index=False))
