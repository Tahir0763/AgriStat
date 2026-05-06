# 🌾 AgriStat — Crop Yield Statistical Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade statistical analytics platform designed for the **Probability & Statistics Semester Project (Spring 2026)**. AgriStat provides deep insights into global agricultural production, climate impacts, and predictive modeling using advanced statistical techniques.

---

## ✨ Key Features

### 📊 Exploratory Data Analysis
- **Global Overview**: Real-time KPI cards tracking dataset health and average yields across 10+ major crop types.
- **Descriptive Statistics**: Detailed profiles including Mean, Median, Variance, Skewness, Kurtosis, and 95% Confidence Intervals.
- **Region Rankings**: Dynamic horizontal bar charts identifying the top-performing countries by yield.

### 📐 Inferential Statistics & Distributions
- **Frequency Distributions**: Histograms with automated Normal Distribution curve fitting.
- **Normality Testing**: Integrated Shapiro-Wilk and Kolmogorov-Smirnov (KS) tests to validate distribution assumptions.
- **Probability Calculator**: Interactive Normal probability engine to calculate $P(X > \text{threshold})$ with Z-score mapping.

### 📉 Advanced Modeling
- **Polynomial Regression**: Model crop yields against environmental variables (Rainfall, Temperature, Pesticides) with adjustable degrees.
- **Inference Engine**: Predict future yields based on custom climate and agricultural inputs.
- **Correlation Matrix**: High-fidelity Pearson correlation heatmap to identify multi-collinearity and feature importance.
- **Climate Heatmap**: 2D interaction analysis showing the synergy between Temperature and Rainfall on global production.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+ / Flask
- **Data Processing**: Pandas, NumPy
- **Scientific Computing**: SciPy (Stats & Probability)
- **Machine Learning**: Scikit-learn (Polynomial Regression, Metrics)
- **Frontend**: 
  - **Logic**: JavaScript (ES6+), Chart.js 4.4
  - **Styling**: Vanilla CSS (Modern Dark Neon Architecture)
  - **Typography**: Outfit, DM Serif Display, JetBrains Mono

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/agristat.git
   cd agristat
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the dashboard**:
   Open your browser and navigate to `http://127.0.0.1:5050`

---

## 📂 Project Structure

```text
.
├── app.py              # Main Flask application & API endpoints
├── yield_df.csv        # Global agricultural dataset
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html      # Single-page modern dashboard UI
└── readme.md           # Documentation
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Developed for the Probability & Statistics Semester Project · Spring 2026**