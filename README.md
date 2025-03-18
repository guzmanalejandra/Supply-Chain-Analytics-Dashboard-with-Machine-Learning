# Supply Chain Analytics Dashboard with Machine Learning

This repository contains a **Streamlit** dashboard for analyzing supply chain data and predicting revenue generated using machine learning. The dashboard loads a CSV dataset containing supply chain metrics and performs the following functions:

- **Data Filtering & KPIs:** Displays key performance indicators such as average revenue, price, products sold, and costs.
- **Data Visualizations:** Offers interactive charts (line charts, scatter plots, histograms, bar charts) to analyze various supply chain metrics like pricing, inventory levels, lead times, sourcing costs, and more.
- **Machine Learning:** Trains a Random Forest Regressor on cleaned numeric data to predict **Revenue generated** based on selected features.
- **Interactive Predictions:** Allows users to input new product parameters to obtain revenue predictions.

---

## Features

- **Interactive Data Filtering:** Filter by product type (and other criteria if needed).
- **KPI Dashboard:** View metrics like average lead time, delivery performance, and customer satisfaction.
- **Multiple Visualizations:** Explore relationships such as Price vs. Revenue, distribution of products sold, and defect rates vs. Revenue.
- **ML Model for Revenue Prediction:**  
  - Cleans and extracts numeric values from columns (e.g., Price, Number of products sold, Stock levels, etc.)  
  - Fills missing values using column means  
  - Trains a Random Forest Regressor and displays the model performance (R² score)  
  - Visualizes actual vs. predicted revenue  
  - Provides interactive input fields for new predictions

---

## Requirements

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Altair](https://altair-viz.github.io/)
- [Scikit-learn](https://scikit-learn.org/)

![image](https://github.com/user-attachments/assets/c3b1640d-d642-4f65-9b7c-40911ed7f5c8)


Install the required packages using pip:

```bash
pip install streamlit pandas numpy altair scikit-learn
