from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


RANDOM_STATE = 42
TEST_MONTHS = 6
FORECAST_MONTHS = 6
FEATURE_COLUMNS = [
	"time_index",
	"month_sin",
	"month_cos",
	"lag_1",
	"lag_3",
	"lag_12",
	"rolling_mean_3",
]


def load_monthly_counts() -> pd.Series:
	"""Aggregate catalog additions into a complete monthly time series."""
	dataset_path = Path(__file__).with_name("Dataset.csv")
	df = pd.read_csv(dataset_path, parse_dates=["date_added"])
	df = df.dropna(subset=["date_added"])
	df["month"] = df["date_added"].dt.to_period("M").dt.to_timestamp()
	monthly_counts = df.groupby("month").size().sort_index()
	all_months = pd.date_range(
		monthly_counts.index.min(), monthly_counts.index.max(), freq="MS"
	)
	return monthly_counts.reindex(all_months, fill_value=0).astype(float)


def build_feature_frame(monthly_counts: pd.Series) -> pd.DataFrame:
	features = pd.DataFrame(index=monthly_counts.index)
	features["count"] = monthly_counts
	features["time_index"] = np.arange(len(features))
	features["month_sin"] = np.sin(2 * np.pi * features.index.month / 12)
	features["month_cos"] = np.cos(2 * np.pi * features.index.month / 12)
	features["lag_1"] = features["count"].shift(1)
	features["lag_3"] = features["count"].shift(3)
	features["lag_12"] = features["count"].shift(12)
	features["rolling_mean_3"] = features["count"].shift(1).rolling(3).mean()
	return features


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
	model.fit(X_train, y_train)
	predictions = np.maximum(model.predict(X_test), 0)
	mae = mean_absolute_error(y_test, predictions)
	rmse = mean_squared_error(y_test, predictions) ** 0.5
	print(f"\n{name}")
	print(f"MAE: {mae:.2f}")
	print(f"RMSE: {rmse:.2f}")
	return {"Model": name, "MAE": mae, "RMSE": rmse, "predictions": predictions}


def forecast_future(model, monthly_counts: pd.Series) -> pd.Series:
	"""Generate recursive forecasts using earlier predictions as future lags."""
	extended_counts = monthly_counts.copy()
	future_index = pd.date_range(
		monthly_counts.index[-1] + pd.offsets.MonthBegin(),
		periods=FORECAST_MONTHS,
		freq="MS",
	)

	for month in future_index:
		extended_counts.loc[month] = np.nan
		features = build_feature_frame(extended_counts)
		prediction = max(float(model.predict(features.loc[[month], FEATURE_COLUMNS])[0]), 0)
		extended_counts.loc[month] = prediction

	return extended_counts.loc[future_index]


def save_forecast_plot(
	monthly_counts: pd.Series,
	test_index: pd.DatetimeIndex,
	test_predictions: np.ndarray,
	future_forecast: pd.Series,
	output_path: Path,
) -> None:
	plt.figure(figsize=(11, 6))
	plt.plot(monthly_counts.index, monthly_counts, label="Actual additions")
	plt.plot(
		test_index,
		test_predictions,
		"o--",
		label="Test predictions",
	)
	plt.plot(
		future_forecast.index,
		future_forecast,
		"s--",
		label="Future forecast",
	)
	plt.axvline(test_index[0], color="gray", linestyle=":", label="Test begins")
	plt.title("Netflix Monthly Content Addition Forecast")
	plt.xlabel("Month")
	plt.ylabel("Titles added")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def main() -> None:
	monthly_counts = load_monthly_counts()
	feature_frame = build_feature_frame(monthly_counts).dropna()
	test_size = min(TEST_MONTHS, max(1, len(feature_frame) // 5))
	train_frame = feature_frame.iloc[:-test_size]
	test_frame = feature_frame.iloc[-test_size:]

	print(
		f"Historical range: {monthly_counts.index[0]:%Y-%m} to "
		f"{monthly_counts.index[-1]:%Y-%m}"
	)
	print(f"Training months: {len(train_frame)} | Test months: {len(test_frame)}")

	models = {
		"Linear Regression": LinearRegression(),
		"Random Forest": RandomForestRegressor(
			n_estimators=200,
			max_depth=8,
			min_samples_leaf=2,
			random_state=RANDOM_STATE,
			n_jobs=-1,
		),
	}
	results = []
	for name, model in models.items():
		results.append(
			evaluate_model(
				name,
				model,
				train_frame[FEATURE_COLUMNS],
				train_frame["count"],
				test_frame[FEATURE_COLUMNS],
				test_frame["count"],
			)
		)

	comparison = pd.DataFrame(results).sort_values("RMSE")
	print("\nForecast model comparison:")
	print(comparison[["Model", "MAE", "RMSE"]].to_string(index=False))

	best_name = comparison.iloc[0]["Model"]
	best_model = models[best_name]
	best_model.fit(feature_frame[FEATURE_COLUMNS], feature_frame["count"])
	future_forecast = forecast_future(best_model, monthly_counts)
	print(f"\nSelected model: {best_name}")
	print("Future monthly predictions:")
	print(future_forecast.round(1).to_string())

	plot_path = Path(__file__).with_name("netflix_trend_forecast.png")
	save_forecast_plot(
		monthly_counts,
		test_frame.index,
		comparison.iloc[0]["predictions"],
		future_forecast,
		plot_path,
	)
	print(f"\nForecast visualization saved to: {plot_path.name}")


if __name__ == "__main__":
	main()
