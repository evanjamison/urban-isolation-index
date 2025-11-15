# scripts/build_tokyo_wards_geojson.py
import json
import os
import glob

INPUT_DIR = "data/external/tokyo_wards_raw"
OUTPUT_FILE = "data/external/jp_tokyo_wards.geojson"

def load_features():
    features = []
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    if len(files) == 0:
        raise FileNotFoundError(f"No ward JSON files found in {INPUT_DIR}")

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Some files include a FeatureCollection
        if data.get("type") == "FeatureCollection":
            features.extend(data["features"])

        # Some files are just a Feature
        elif data.get("type") == "Feature":
            features.append(data)

        else:
            raise ValueError(f"Unknown GeoJSON structure in {fpath}")

    return features


def main():
    features = load_features()

    merged = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {OUTPUT_FILE} with {len(features)} features")


if __name__ == "__main__":
    main()
