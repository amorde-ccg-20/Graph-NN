import argparse
from pathlib import Path

from q3.data_processing.load import load_flow
from q3.data_processing.process import resample_flow, save_flow
from q3.visualization.weekly_flow import plot_week


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-2020", type=Path, required=True)
    parser.add_argument("--year-2021", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--aggregation", choices=("mean", "sum"), required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    processed = root / "dataset" / "processed"
    figures = root.parent / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    years = {}
    for year, path in ((2020, args.year_2020), (2021, args.year_2021)):
        original = load_flow(path, args.metadata, year)
        resampled = resample_flow(original, args.aggregation)
        print(f"{year}: {len(original)} five-minute rows, {original.shape[1]} sensors")
        print(f"  original missing values: {original.isna().sum().sum():,}")
        print(f"  15-minute rows: {len(resampled)}; missing values: {resampled.isna().sum().sum():,}")
        save_flow(processed / f"sd_flow_{year}_15min.npz", resampled)
        years[year] = resampled

    if not years[2020].columns.equals(years[2021].columns):
        raise ValueError("Sensor IDs or their order differ between the two years")

    figure_path = figures / "q3_weekly_traffic.pdf"
    plot_week(years[2020], figure_path)
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
