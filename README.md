# CropFusionNet: An Interpretable Deep Learning Framework for Probabilistic Crop Yield Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository contains the implementation and comprehensive evaluation of **CropFusionNet**, an interpretable deep learning framework specifically designed for crop yield forecasting in Germany. CropFusionNet integrates multi-modal geospatial data (satellite imagery, climate, soil, and topography) with advanced neural architectures to deliver accurate and explainable yield predictions at multiple lead times.

### Key Innovations

- **CropFusionNet Architecture**: Novel multi-modal fusion mechanism for seamless integration of remote sensing, climate, soil, and topographic data
- **Interpretability by Design**: Built-in explainability through attention mechanisms and feature importance analysis
- **Comprehensive Evaluation Framework**: Covers predictive performance, spatial error distribution, lead-time analysis, sensitivity analysis, and ablation studies
- **Multi-Lead-Time Forecasting**: Generate predictions at multiple lead times from months to weeks before harvest
- **Crop-Specific Models**: Dedicated implementations for major German crops (winter wheat, winter barley, silage maize, winter rapeseed, winter rye)

## Project Structure

```
crop-yield-forecasting-germany/
├── README.md                          # This file
├── .gitignore                         # Git ignore patterns
├── .pre-commit-config.yaml            # Pre-commit hooks configuration
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 00_exploration/                # Data exploration notebooks
│   ├── 01_preprocessing/              # Data preprocessing pipeline
│   ├── 02_feature_engineering/        # Feature engineering
│   ├── 03_modeling/                   # Model training
│   └── 04_evaluation/                 # Model evaluation and analysis
│
└── src/                               # Source code
    ├── config/                        # Configuration files per crop
    ├── dataset/                       # Dataset handling classes
    ├── loss/                          # Custom loss functions
    ├── models/                        # Model architectures
    │   ├── CropFusionNet/             # Main proposed architecture
    │   ├── VanillaLSTM/               # Baseline LSTM
    │   ├── SimpleTransformer/         # Transformer-based model
    │   └── ResCNN/                    # ResNet-CNN architecture
    ├── scaler/                        # Pre-trained scalers for normalization
    ├── train/                         # Training notebooks
    ├── test/                          # Testing notebooks
    └── utils/                         # Utility functions

Note: Data, output, and temporary files are excluded from version control.
```

## Data Sources

The framework integrates data from multiple sources:

- **Remote Sensing**: Sentinel-2 optical and Sentinel-1 SAR data
  - Indices: NDVI, EVI, FPAR, LAI
- **Climate Data**: Daily meteorological observations
  - Variables: Temperature (min/avg/max), Precipitation, Radiation, VPD, Evapotranspiration, Soil moisture, Soil temperature, Sunshine duration, Climatic water balance
- **Soil Data**: Harmonized soil properties
  - Variables: Soil quality index, Soil depth, Texture
- **Topography**: Digital elevation models
  - Variables: Elevation, Slope
- **Irrigation**: Irrigation extent and management
- **Phenology**: Crop development stage information

- **Yield**: Observed crop yields for training and validation

## Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA support (recommended for training)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/crop-yield-forecasting-germany.git
cd crop-yield-forecasting-germany
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

The project follows a sequential pipeline to prepare data and train CropFusionNet:

#### 1. Data Preparation

Prepare multi-modal geospatial data from various sources:

```bash
jupyter notebook notebooks/01_preprocessing/
```

Execute notebooks in order:

- `00_phenology_data_preparation.ipynb` - Process crop phenology data
- `01_climate_data_preparation.ipynb` - Prepare daily climate variables
- `03_static_data_preparation.ipynb` - Extract static soil and topographic features
- `04_remote_sensing_data_preparation.ipynb` - Process satellite-derived indices (NDVI, EVI, FPAR, LAI)
- `05_check_for_valid_data_points.ipynb` - Quality control and validation
- `06_timeseries_data_preparation.ipynb` - Organize data into time series format
- `07_calculate_scalers.ipynb` - Compute normalization parameters

#### 2. CropFusionNet Training

Train CropFusionNet for your target crop:

See training notebooks in `src/train/` for detailed instructions on:

- Configuring the model for specific crops (winter wheat, winter barley, silage maize, etc.)
- Setting hyperparameters
- Training with multi-modal data
- Saving trained models

#### 3. Model Evaluation & Interpretation

Evaluate CropFusionNet performance and extract insights:

Execute evaluation notebooks in `notebooks/04_evaluation/`:

- `04.1_predictive_performance_&_benchmarking.ipynb` - Benchmark CropFusionNet against baselines
- `04.2_spatial_distribution_of_prediction_error.ipynb` - Analyze geographic error patterns
- `04.3_forecast_skills_at_different_lead_times.ipynb` - Assess forecast skill at different lead times
- `04.4_interpretability_of_environmental_variables.ipynb` - Interpret feature importance using attention weights
- `04.5_sensitivity_to_extreme_climate_events.ipynb` - Test model robustness to extreme weather
- `04.6_ablation_study_of_model_components.ipynb` - Analyze contribution of CropFusionNet components

## CropFusionNet Architecture

**CropFusionNet** is the core contribution of this work, designed specifically for crop yield forecasting. The architecture features:

### Core Components

1. **Multi-Modal Data Encoder**
   - Separate pathways for remote sensing, climate, soil, and topographic data
   - Specialized preprocessing for each data modality

2. **Temporal Encoding with Attention**
   - Attention-based LSTM for capturing temporal dependencies
   - Multi-head attention mechanism to weight important timesteps and features

3. **Feature Fusion Module**
   - Adaptive fusion of time-varying (remote sensing, climate) and static (soil, topography) features

4. **Interpretability Layer**
   - Feature importance estimation through variable selection network

### Key Advantages Over Baselines

- Superior predictive accuracy compared to vanilla LSTM, Transformers, and CNN baselines
- Built-in interpretability through attention mechanisms
- Effective handling of multi-modal data with varying temporal patterns
- Robust to extreme weather events

For implementation details, see `src/models/CropFusionNet/model.py`.

## Comparison with Baseline Models

The repository includes baseline implementations for comparison:

- **VanillaLSTM**: Standard LSTM for temporal sequence modeling
- **SimpleTransformer**: Transformer-based architecture
- **ResCNN**: Residual CNN with temporal encoding

These baselines demonstrate the superior performance and interpretability of CropFusionNet.

## Configuration

Crop-specific configurations are in `src/config/`:

- `winter_wheat.py` - Winter wheat yield forecasting
- `winter_barley.py` - Winter barley yield forecasting
- `silage_maize.py` - Silage maize yield forecasting

Each configuration specifies:

- Data paths and train/test/validation split definitions
- Feature selection (soil, topography, irrigation, remote sensing, climate)
- Scaling parameters for normalization
- Model hyperparameters and training settings

## Results

### Evaluation Notebooks

The repository includes comprehensive evaluation in `notebooks/04_evaluation/`:

- `04.1_predictive_performance_&_benchmarking.ipynb` - CropFusionNet vs. baseline model performance
- `04.2_spatial_distribution_of_prediction_error.ipynb` - Geographic patterns in prediction accuracy
- `04.3_forecast_skills_at_different_lead_times.ipynb` - Forecast skill decay with increasing lead time
- `04.4_interpretability_of_environmental_variables.ipynb` - Feature importance and model explainability
- `04.5_sensitivity_to_extreme_climate_events.ipynb` - Model robustness to extreme weather
- `04.6_ablation_study_of_model_components.ipynb` - Contribution of individual CropFusionNet components

<!-- ### Key Results

- CropFusionNet achieves significant improvements over baseline models across all crops
- Multi-lead-time forecasting enables harvest outcome prediction up to [X] months in advance
- Attention-based interpretability provides actionable insights into yield-driving factors
- Robust performance across diverse growing conditions and weather scenarios -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- [Krishnagopal Halder](https://github.com/geonextgis)

## Acknowledgments

<!-- - Data sources: [List relevant data providers]
- Funding: [List funding sources if applicable]
- Computing resources: [ZALF/HPC facilities if used] -->
