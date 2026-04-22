# Migrated bio_age reference summary

This is a summary-level migration of the previous example results. Re-run the benchmark to generate feature_sets/ and figures with current bio_age column names.

## Main reference axes

| bio_age_axis | model | n_features | sample_gap_mae | subject_gap_mae | subject_gain | bio_age_subject_mae | bio_age_vs_true_corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bio_age_ei_texture | elasticnet | 168 | 2.8241 | 2.5236 | 6.7453 | 9.3821 | 0.5944 |
| bio_age_texture | elasticnet | 144 | 3.2717 | 2.8664 | 6.4025 | 9.3690 | 0.5923 |
| bio_age_ei | elasticnet | 12 | 5.9078 | 5.5476 | 3.7213 | 10.8407 | 0.5477 |
| bio_age_ei | elasticnet | 1 | 6.1943 | 5.8087 | 3.4602 | 10.9933 | 0.5430 |
| bio_age_ei_texture | ridge | 168 | 2.9686 | 2.6807 | 6.5882 | 9.4334 | 0.5444 |
| bio_age_texture | ridge | 144 | 3.2406 | 2.8936 | 6.3753 | 9.4297 | 0.5430 |
| bio_age_ei | ridge | 12 | 5.9384 | 5.5691 | 3.6998 | 10.8407 | 0.5522 |
| bio_age_ei | ridge | 1 | 6.1937 | 5.8077 | 3.4612 | 10.9919 | 0.5430 |
