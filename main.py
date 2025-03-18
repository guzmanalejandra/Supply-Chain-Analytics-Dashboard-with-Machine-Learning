import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import re


### Cleaning 

def extract_number(x):
    """
    Attempts to extract a numeric value from a string.
    Removes any characters except digits, decimal points, minus signs.
    """
    if isinstance(x, str):
        # Remove any characters that are not digits, decimal point, or minus sign.
        x = re.sub(r'[^\d\.-]', '', x)
    try:
        return float(x)
    except:
        return np.nan

##############################
# 1) PAGE TITLE & DESCRIPTION
##############################
st.title("Supply Chain Analytics Dashboard with Machine Learning")

st.markdown("""
This dashboard analyzes various supply chain metrics such as:
- **Pricing, Sales & Revenue**
- **Inventory & Lead Times**
- **Shipping & Sourcing Costs**
- **Quality & Defect Rates**

It also uses machine learning to predict **Revenue generated** based on key numeric supply chain features.

Use the sidebar to filter data by product type.
---
""")

##############################
# 2) DATA LOADING & FILTERING
##############################
@st.cache_data
def load_data():
    df = pd.read_csv("supply_chain_data.csv")
    return df

df = load_data()

# Sidebar filter: Product Type (if available)
if "Product type" in df.columns:
    product_types = df["Product type"].unique().tolist()
    selected_types = st.sidebar.multiselect("Select Product Type", product_types, default=product_types)
    df = df[df["Product type"].isin(selected_types)]

##############################
# 3) KPI DISPLAY
##############################
st.header("Key Performance Indicators (KPIs)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    avg_revenue = df["Revenue generated"].mean() if "Revenue generated" in df.columns else np.nan
    st.metric("Average Revenue", f"${avg_revenue:,.2f}")
with col2:
    avg_price = df["Price"].mean() if "Price" in df.columns else np.nan
    st.metric("Average Price", f"${avg_price:,.2f}")
with col3:
    avg_sold = df["Number of products sold"].mean() if "Number of products sold" in df.columns else np.nan
    st.metric("Avg. Products Sold", f"{avg_sold:.0f}")
with col4:
    avg_costs = df["Costs"].mean() if "Costs" in df.columns else np.nan
    st.metric("Average Costs", f"${avg_costs:,.2f}")

st.write("---")

##############################
# 4) DATA VISUALIZATIONS
##############################
st.header("Supply Chain Data Visualizations")

# Chart 1: Scatter Plot – Price vs. Revenue Generated
if "Price" in df.columns and "Revenue generated" in df.columns:
    scatter_price_revenue = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("Price:Q", title="Price"),
        y=alt.Y("Revenue generated:Q", title="Revenue Generated"),
        tooltip=["Product type", "SKU", "Price", "Revenue generated"]
    ).properties(
        title="Price vs. Revenue Generated",
        width=600,
        height=300
    )
    st.altair_chart(scatter_price_revenue, use_container_width=True)

# Chart 2: Bar Chart – Average Revenue by Product Type
if "Product type" in df.columns and "Revenue generated" in df.columns:
    revenue_by_type = df.groupby("Product type")["Revenue generated"].mean().reset_index()
    bar_revenue_type = alt.Chart(revenue_by_type).mark_bar().encode(
        x=alt.X("Product type:N", title="Product Type"),
        y=alt.Y("Revenue generated:Q", title="Avg. Revenue Generated"),
        tooltip=["Product type", "Revenue generated"]
    ).properties(
        title="Average Revenue by Product Type",
        width=600,
        height=300
    )
    st.altair_chart(bar_revenue_type, use_container_width=True)

# Chart 3: Histogram – Number of Products Sold
if "Number of products sold" in df.columns:
    hist_products_sold = alt.Chart(df).mark_bar().encode(
        x=alt.X("Number of products sold:Q", bin=alt.Bin(maxbins=30), title="Number of Products Sold"),
        y="count()"
    ).properties(
        title="Distribution of Products Sold",
        width=600,
        height=300
    )
    st.altair_chart(hist_products_sold, use_container_width=True)

# Chart 4: Scatter Plot – Defect Rates vs. Revenue Generated
if "Defect rates" in df.columns and "Revenue generated" in df.columns:
    scatter_defects = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("Defect rates:Q", title="Defect Rates"),
        y=alt.Y("Revenue generated:Q", title="Revenue Generated"),
        tooltip=["SKU", "Defect rates", "Revenue generated"]
    ).properties(
        title="Defect Rates vs. Revenue Generated",
        width=600,
        height=300
    )
    st.altair_chart(scatter_defects, use_container_width=True)

st.write("---")

##############################
# 5) MACHINE LEARNING: PREDICT REVENUE GENERATED
##############################
st.header("Machine Learning: Predict Revenue Generated")

st.markdown("""
We will train a **Random Forest Regressor** to predict **Revenue generated** based on the following selected numeric features:

- Price  
- Number of products sold  
- Stock levels  
- Order quantities  
- Shipping costs  
- Manufacturing costs  
- Defect rates  
- Costs  

**Note:** Only these columns (after cleaning) will be used.
""")

# Define a focused list of feature columns expected to have numeric values
feature_cols = [
    "Price", "Number of products sold", "Stock levels", "Order quantities", 
    "Shipping costs", "Manufacturing costs", "Defect rates", "Costs"
]

# Check if these columns exist in the dataset
available_features = [col for col in feature_cols if col in df.columns]

# Create a copy for ML using the selected features and target "Revenue generated"
ml_df = df[available_features + ["Revenue generated"]].copy()

# Convert the selected feature columns to numeric using our helper function,
# and fill missing values with the column mean.
cols_to_remove = []
for col in available_features:
    ml_df[col] = ml_df[col].apply(extract_number)
    if ml_df[col].notna().sum() == 0:
        st.warning(f"Column '{col}' contains no valid numeric values and will be removed from the features.")
        cols_to_remove.append(col)
    else:
        ml_df[col] = ml_df[col].fillna(ml_df[col].mean())

# Update available_features by removing any columns with no valid data
available_features = [col for col in available_features if col not in cols_to_remove]

# Process target column "Revenue generated"
ml_df["Revenue generated"] = ml_df["Revenue generated"].apply(extract_number)
if ml_df["Revenue generated"].notna().sum() == 0:
    st.error("The target column 'Revenue generated' has no valid numeric values.")
else:
    ml_df["Revenue generated"] = ml_df["Revenue generated"].fillna(ml_df["Revenue generated"].mean())

# Drop any remaining rows with missing values (should be minimal after filling)
ml_df = ml_df.dropna()

if ml_df.empty or len(available_features) == 0:
    st.error("No valid data available for ML after cleaning. Please check your dataset and ensure that the numeric columns contain valid data.")
else:
    X = ml_df[available_features]
    y = ml_df["Revenue generated"]

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    from sklearn.ensemble import RandomForestRegressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=123)
    rf_reg.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = rf_reg.predict(X_test)
    r2 = rf_reg.score(X_test, y_test)
    st.write("### Model Performance")
    st.write(f"R² score on test set: {r2:.2f}")

    # Scatter plot: Actual vs. Predicted Revenue
    import altair as alt
    test_df_ml = X_test.copy()
    test_df_ml["Actual Revenue"] = y_test
    test_df_ml["Predicted Revenue"] = y_pred
    scatter_ml = alt.Chart(test_df_ml.reset_index()).mark_circle(size=60).encode(
        x=alt.X("Actual Revenue:Q", title="Actual Revenue"),
        y=alt.Y("Predicted Revenue:Q", title="Predicted Revenue"),
        tooltip=["Actual Revenue", "Predicted Revenue"]
    ).properties(
        title="Actual vs. Predicted Revenue Generated",
        width=600,
        height=300
    )
    st.altair_chart(scatter_ml, use_container_width=True)

    st.write("---")
    st.subheader("Predict Revenue Generated for New Input")
    st.markdown("Adjust the parameters below to predict the revenue for a new product:")

    new_data = {}
    for col in available_features:
        default_val = float(ml_df[col].mean())
        new_data[col] = st.number_input(f"{col}", value=default_val)

    new_features = pd.DataFrame(new_data, index=[0])
    prediction_ml = rf_reg.predict(new_features)[0]
    st.write(f"**Predicted Revenue Generated:** ${prediction_ml:,.2f}")
