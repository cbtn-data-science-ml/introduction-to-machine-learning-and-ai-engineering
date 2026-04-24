# Introduction to Machine Learning and AI Engineering

Repo for the **Introduction to Machine Learning and AI Engineering** course from **CBT Nuggets** with Jonathan Barrios.

This repo contains the small datasets and helper utilities used throughout the course notebooks. The goal is to keep the notebooks focused on modeling, training, visualization, and reasoning about machine learning workflows.

---

## What’s in this repo

### Datasets

#### `clean_autonomous_taxi_rides.csv`
A small clean tabular dataset used in the early PyTorch notebooks.

Columns:
- `trip_distance_miles`
- `pickup_hour`
- `passenger_count`
- `ride_duration_min`

Good for:
- linear regression
- logistic regression
- small neural network intro examples
- feature selection and target engineering

#### `delivery_drones.csv`
A small delivery dataset used for challenge notebooks and transfer practice.

Columns:
- `delivery_distance_miles`
- `departure_hour`
- `payload_weight_lb`
- `flight_duration_min`

Good for:
- challenge notebooks
- regression practice
- feature engineering
- comparing workflows across domains

---

## Helper utility

### `helper_utils.py`

This file contains lightweight plotting helpers used in some notebooks so the notebook cells can stay focused on the modeling logic instead of repeating visualization boilerplate.

Current public helpers:
- `plot_points(...)`
- `plot_fit(...)`
- `plot_training_progress(...)`

These helpers are intentionally limited to plotting and display logic.

They do **not** hide:
- normalization
- de-normalization
- model definition
- training loops
- business decision logic

That is intentional. In this course, the notebooks should still show the actual ML workflow clearly.

---

## Install requirements

Most notebooks in this repo use:

```python
torch
matplotlib
pandas
IPython
