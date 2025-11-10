Continue = 'Stop'
python -m pip install -r requirements.txt
python -m src.cli.01_ingest_jp_estat --year 2020
python -m src.cli.01_ingest_jp_tokyo
python -m src.cli.01_ingest_us_acs --state 36 --year 2020
python -m src.cli.02_features_jp
python -m src.cli.02_features_us --msa NYC
python -m src.cli.03_build_index
python -m src.cli.04_validate_spatial
python -m src.cli.05_ml_pca
python -m src.cli.05_ml_cluster
