# Integrating the sleep models with the ring and the app

Pipeline `0.1.1` · Feature dictionary v2 · 18 Nov 2026

This is the app-side companion to `README.md`. The README says what the models are; this
says how to feed them from the ring. The Python reference for everything here is
`src/wualt_sleep/data/ring_row.py`; the native port is checked against
`examples/ring_row_vectors.json`.

## 1. What the app sends

| when | what | notes |
|---|---|---|
| session start | `SleepPipeline(..., bedtime=...)` | **Pass `bedtime`.** It arms the 6-hour fallback; without it the fallback never fires and a night that never crosses 0.7 produces no record. |
| every 15 minutes | `push_row(Row, accel_epochs=...)` | `Row.values` = the **32 shared features** in dictionary order (not 36). `accel_epochs` = the (30, 13) block, or `None` if the accelerometer produced nothing. |
| when `out.onset_event` fires | set `NightState.onset_at = event.confirmed_at` | Rows starting from then on carry the onset-relative features. |
| wake-up | `finalise(wake_time=..., history_nights=n)` | `history_nights` = nights already recorded; under 14 → `sleep_score` is null. |

The four `layer1`-scope dictionary entries (`hours_since_bedtime`, `sustained_still_rows`,
`row_evening_hr_delta`, `hours_since_record_start`) are **not** in the row. No shipped
model reads three of them, and the pipeline keeps `sustained_still_rows` itself.

## 2. Which model reads what

| block | features | Layer 1 (onset) | Layer 2 (staging) |
|---|---|---|---|
| heart rate | `hr_mean hr_min hr_std hr_slope hr_delta_vs_rhr` | yes | yes |
| HRV | `rmssd sdnn pnn50 poincare_sd1 poincare_sd2 sd1_sd2_ratio hrv_sample_entropy` | yes | yes |
| motion | `activity_count_mean activity_count_max still_fraction acc_magnitude_std posture_change_count movement_burst_count` | yes | yes |
| temperature | `temp_mean temp_slope temp_std temp_delta_vs_base` | yes | **no** |
| time | `clock_hour_sin clock_hour_cos minutes_in_record minutes_since_onset cycle_phase_sin cycle_phase_cos` | **no** | yes |
| quality | `sqi_ppg hrv_artifact_rate sqi_acc` | yes | yes |
| baseline | `baseline_confidence` | yes | yes |

The WASO channel reads only the (30, 13) accelerometer block.

## 3. How each row value is built

**Confirmed** means it is stated in this package or forced by the shipped artifacts'
training statistics. **Provisional** means it is the best reconstruction available. The
training-time row builders (`data/dreamt.py`, `data/bidsleep.py`) are not in this repo; when
they are published, provisional definitions are replaced and the row layout does not change.

| feature | definition | status |
|---|---|---|
| cardiac window | the 90 s PPG burst only (`ppg_burst_seconds: 90`) | confirmed |
| beats | pass **every** interval the ring reported, in order; `clean_nn` splits missed beats, applies 0.3–2.0 s and the 20 % rule, and never differences across a rejected beat | confirmed |
| `hr_mean` | mean of 60/NN over accepted beats (60 / mean NN differs by < 0.3 bpm) | confirmed |
| `hr_min` | minimum per-beat HR over the burst | provisional |
| `hr_std` | SD of per-beat HR, ddof 1 (ddof changes it 0.6 % at ~90 beats) | confirmed |
| `hr_slope` | least-squares slope of per-beat HR on beat time, × 3600 → bpm/hr. Within the burst is forced: training SD is 275 bpm/hr | provisional (method) |
| `hr_delta_vs_rhr` | `hr_mean` − reference. Reference = personal resting HR when given, else mean `hr_mean` of the session's first two rows (training used a within-night pre-sleep reference) | provisional |
| HRV block | `hrv.hrv_features`; NaN below 40 valid beats | confirmed |
| motion block | `motion.motion_features_on_grid` on the 50 Hz grid | confirmed |
| `temp_mean` | mean of the row's readings, each reading = mean of usable sensors | provisional |
| `temp_slope` | least-squares slope over the row's readings, °C/hr (≥ 3 readings) | provisional |
| `temp_std` | SD of the row's readings, ddof 1 | provisional |
| `temp_delta_vs_base` | `temp_mean` − reference, same rule as `hr_delta_vs_rhr` | provisional |
| `clock_hour_*` | sin/cos(2π·h/24), h = local fractional hour at row start | provisional |
| `minutes_in_record` | minutes from session start to row start (training: lab recording start) | provisional |
| `minutes_since_onset` | minutes from `OnsetEvent.confirmed_at` to row start; NaN before it | provisional (see §6) |
| `cycle_phase_*` | sin/cos(2π·(minutes_since_onset mod 90)/90); NaN with it | confirmed |
| `sqi_ppg` / `hrv_artifact_rate` / `sqi_acc` | `beat_coverage`, `NNSeries.artifact_rate`, `time_coverage` | confirmed |
| `baseline_confidence` | min(1, (n − 1)/13), n = night number (0 on night 1, 1.0 from night 14) | provisional (see §6) |

**Missing values.** NaN is legal; never impute. The yaml's numeric policies apply:
burst and posture counts → 0, `sqi_ppg` → 0, `hrv_artifact_rate` → 1, `sqi_acc` → 0,
`baseline_confidence` → 0. **Do not** apply the per-cycle-phase baseline subtraction
(`baseline_subtract` in the yaml): no shipped model was trained with it — the Layer 2
`mu` values are absolute levels and neither corpus has cycle-phase data.

## 4. The accelerometer

- Units **g** (divide mg by 1000). Every threshold in the package is in g.
- Resample linearly onto the **50 Hz** grid (`GRID_HZ`); interpolate across gaps, never drop
  them; a grid point is covered when a real sample is within 1 s.
- Compute the epoch block with **one 30 s epoch of look-back**, then drop it
  (`ring_row.accel_epochs_for_row`), so the first epoch's `posture_delta` is measured against
  the previous epoch as it was in training.
- **Native rate.** The models were trained on 64 Hz (DREAMT) and 50–100 Hz (BIDSleep)
  wrist devices. Interpolating a 12.5 Hz stream to 50 Hz cannot recreate the 3–10 Hz band,
  and jerk and activity read low. Run the accelerometer at ≥ 50 Hz overnight, or validate
  the models on training data downsampled to the ring's rate before shipping.

## 5. Calling the models (unchanged)

See README §4. The wake channel must be given a **whole night** of epochs in one call;
`SleepPipeline` does this from 0.1.1 (it scored row by row before — see Changes).

## 6. Open decisions for ML

| item | why it matters |
|---|---|
| Onset timestamp | `confirmed_at` is the end of the **second** row above 0.7; Layer 1's accuracy was scored on the **first**. The emitted onset is one row later than the scored one. Back-dating moves `onset_time`, duration and the staging buffer. |
| `baseline_confidence` range | Training mean 0.12 (SD 0.13; always 0 on DREAMT). From night 14 the app sends 1.0, ~6.8 SD outside Layer 2's training range. Test, or retrain with multi-night data. |
| Lab-schedule features | `minutes_in_record` and the clock pair encode the lab schedule; removing all three cost 0.014 kappa. Consider retraining Layer 2 without them. |
| Training table | Layer 1 meta names `unified_g50_trimmed.parquet`; `default.yaml` describes `unified_duty90.parquet`; Layer 2 meta names none. |
| Provisional rows in §3 | Publish `data/dreamt.py` and `data/bidsleep.py` (and the pipelines) to settle them. |

## 7. Parity vectors

`examples/ring_row_vectors.json` holds four synthetic rows (awake, asleep after onset,
missing PPG and temperature with an accelerometer gap, a turn-over) with raw inputs and the
expected 32 row values and (30, 13) epoch block. The accelerometer is given as an exact
recipe, documented at the top of `examples/make_ring_vectors.py`. Tolerance 1e-6 relative.

## Changes in 0.1.1

- **Wake channel scored per night, not per row** (`runtime/pipeline.py`). The scorer's
  context and `act_mean_rel` were fitted per night; per-row scoring divided by the row's own
  median. On a synthetic night with 30 minutes of waking movement, the shipped ONNX scorer
  called 0 of 60 epochs wake per row and 26 of 60 per night; still periods were unchanged.
- **`cold_start` now clears** once `baseline_confidence` reaches 1.0 (the context copy was
  never updated).
- **Docs match the code**: one 0.7 onset gate (not 0.5 and 0.7); what `confirmed_at` is.
- **Added** `data/ring_row.py`, `examples/`, this file.
