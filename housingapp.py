import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# ==============================
# PATHS
# ==============================
BASE = os.path.dirname(__file__)
RAW_CSV = os.path.join(BASE, "housingprice.csv")
CLEAN_CSV = os.path.join(BASE, "housingprice_cleaned.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLF_PATH = os.path.join(MODEL_DIR, "investment_model.pkl")
REG_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
COL_PATH = os.path.join(MODEL_DIR, "model_columns.pkl")

# ==============================
# CLEANING FUNCTIONS
# ==============================
def parse_count(x):
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return np.nan

def clean_csv(inpath, outpath):
    df = pd.read_csv(inpath)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    num_cols = [
        "bhk","size_in_sqft","price_in_lakhs","age_of_property",
        "nearby_schools","nearby_hospitals"
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_count)

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna("N/A", inplace=True)

    df.to_csv(outpath, index=False)
    return df

# ==============================
# LOAD DATA
# ==============================
if not os.path.exists(CLEAN_CSV):
    df = clean_csv(RAW_CSV, CLEAN_CSV)
else:
    df = pd.read_csv(CLEAN_CSV)

# ==============================
# FEATURE ENGINEERING
# ==============================
df["price_per_sqft"] = (df["price_in_lakhs"] * 100000) / df["size_in_sqft"]
df["amenities_count"] = df["amenities"].astype(str).apply(
    lambda x: len([i for i in x.split(",") if i.strip()])
)

# Target labels
city_median = df.groupby("city")["price_per_sqft"].transform("median")

df["good_investment"] = (
    (df["price_per_sqft"] <= city_median) &
    (df["age_of_property"] <= 10) &
    (df["nearby_schools"] >= 2) &
    (df["amenities_count"] >= 3)
).astype(int)

df["future_price_5y"] = df["price_in_lakhs"] * (1.1 ** 5)

# ==============================
# ENCODING
# ==============================
df_ml = df[[
    "city","bhk","size_in_sqft","price_in_lakhs",
    "age_of_property","nearby_schools",
    "amenities_count","price_per_sqft",
    "good_investment","future_price_5y"
]]

df_encoded = pd.get_dummies(df_ml, drop_first=True)

X = df_encoded.drop(columns=["good_investment","future_price_5y"])
y_class = df_encoded["good_investment"]
y_reg = df_encoded["future_price_5y"]

# ==============================
# TRAIN MODELS (ONLY ONCE)
# ==============================
if not (os.path.exists(CLF_PATH) and os.path.exists(REG_PATH)):
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42
    )
    clf.fit(X, y_class)

    reg = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42
    )
    reg.fit(X, y_reg)

    joblib.dump(clf, CLF_PATH)
    joblib.dump(reg, REG_PATH)
    joblib.dump(X.columns, COL_PATH)

# ==============================
# LOAD MODELS
# ==============================
clf = joblib.load(CLF_PATH)
reg = joblib.load(REG_PATH)
model_cols = joblib.load(COL_PATH)

# ==============================
# STREAMLIT UI
# ==============================
st.title("🏠 Real Estate Investment Advisor")

menu = st.sidebar.radio("Navigation", ["Introduction", "EDA", "Prediction"])

if menu == "Introduction":
    st.write("""
    ### 📌 Project Features
    **This application assists real estate investor by:** 
    - Classifying whether a property is a Good Investment 
    - Predicating the Estimated Property price After 5 Years 
    - Providing interactive EDA visualizations
     Supporting intelligent decision-making for buyers ,sellers, and investors
    """)
    
    st.subheader("Skills Used")
    st.markdown( """ Python,EDA,Machine Learning,Streamlit,Feature Engineering,MLflow
     """ )
elif menu == "EDA":
    
    st.subheader("1.Price Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df["price_in_lakhs"].dropna(), bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Property Prices (Lakhs)")
    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 2. Distribution of Property Sizes
    st.subheader("2. Distribution of Property Sizes")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df["size_in_sqft"].dropna(), bins=30, color="lightgreen", edgecolor="black")
    ax.set_title("Distribution of Property Sizes (sq ft)")
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 3. Average Price by City
    st.subheader("3️⃣ Average Price by City")
    avg_price_city = df.groupby("city")["price_in_lakhs"].mean().sort_values()
    fig, ax = plt.subplots()
    ax.barh(avg_price_city.index, avg_price_city.values)
    st.pyplot(fig)
    
     # 4. Amenities vs Price
    st.subheader("4️⃣ Amenities Count vs Price")
    amenities_price = df.groupby("amenities_count")["price_in_lakhs"].mean()
    fig, ax = plt.subplots()
    ax.bar(amenities_price.index, amenities_price.values)
    st.pyplot(fig)
    
    
    # 5. Correlation Heatmap
    st.subheader("5️⃣ Correlation Matrix")
    corr = df.select_dtypes(include=["int64","float64"]).corr()
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)
    
     # 6. Average Price per Sq Ft by State
    st.subheader("6. Average Price per Sq Ft by State")
    avg_price_sqft_state = df.groupby("state")["price_per_sqft"].mean().dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(avg_price_sqft_state.index, avg_price_sqft_state.values, color="purple")
    ax.set_title("Average Price per Sq Ft by State")
    ax.set_xlabel("State")
    ax.set_ylabel("Price per Sq Ft")
    ax.set_xticklabels(avg_price_sqft_state.index, rotation=45)
    st.pyplot(fig)

    # 7. Average Property Price by City
    st.subheader("7. Average Property Price by City")
    avg_price_city = df.groupby("city")["price_in_lakhs"].mean().dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(avg_price_city.index, avg_price_city.values, color="teal")
    ax.set_title("Average Property Price by City (Lakhs)")
    ax.set_xlabel("City")
    ax.set_ylabel("Average Price (Lakhs)")
    ax.set_xticklabels(avg_price_city.index, rotation=45)
    st.pyplot(fig)

    # 8. Median Age of Properties by Locality
    st.subheader("8. Median Age of Properties by Locality")
    median_age_locality = df.groupby("locality")["age_of_property"].median().dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(median_age_locality.head(10).index, median_age_locality.head(10).values, color="orange")
    ax.set_title("Median Age of Properties by Locality (Top 10)")
    ax.set_xlabel("Locality")
    ax.set_ylabel("Median Age (Years)")
    ax.set_xticklabels(median_age_locality.head(10).index, rotation=45)
    st.pyplot(fig)

    # 9. BHK Distribution Across Cities
    st.subheader("9. BHK Distribution Across Cities")
    bhk_city_distribution = df.groupby(["city","bhk"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10,6))
    bhk_city_distribution.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("BHK Distribution Across Cities")
    ax.set_xlabel("City")
    ax.set_ylabel("Number of Properties")
    ax.legend(title="BHK")
    st.pyplot(fig)

    # 10. Top 5 Most Expensive Localities
    st.subheader("10. Price Trends for Top 5 Most Expensive Localities")
    top_localities = df.groupby("locality")["price_per_sqft"].mean().dropna().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(top_localities.index, top_localities.values, color="crimson")
    ax.set_title("Top 5 Most Expensive Localities (Avg Price per Sq Ft)")
    ax.set_xlabel("Locality")
    ax.set_ylabel("Average Price per Sq Ft")
    ax.set_xticklabels(top_localities.index, rotation=45)
    st.pyplot(fig)

    # 11. Correlation Matrix
    st.subheader("11. Correlation Between Numeric Features")
    numeric_df = df.select_dtypes(include=["int64","float64"])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10,6))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Matrix", pad=20)
    st.pyplot(fig)

    # 12. Nearby Schools vs Price per Sq Ft
    st.subheader("12. Nearby Schools vs Price per Sq Ft")
    fig, ax = plt.subplots(figsize=(6,4))
    yes_schools = df[df["nearby_schools"]=="Yes"]["price_per_sqft"].dropna()
    no_schools = df[df["nearby_schools"]=="No"]["price_per_sqft"].dropna()
    ax.hist(yes_schools, bins=20, alpha=0.6, label="Yes", color="green")
    ax.hist(no_schools, bins=20, alpha=0.6, label="No", color="red")
    ax.set_title("Price per Sq Ft Distribution by Nearby Schools")
    ax.set_xlabel("Price per Sq Ft")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # 13. Nearby Hospitals vs Price per Sq Ft
    st.subheader("13. Nearby Hospitals vs Price per Sq Ft")
    fig, ax = plt.subplots(figsize=(6,4))
    yes_hospitals = df[df["nearby_hospitals"]=="Yes"]["price_per_sqft"].dropna()
    no_hospitals = df[df["nearby_hospitals"]=="No"]["price_per_sqft"].dropna()
    ax.hist(yes_hospitals, bins=20, alpha=0.6, label="Yes", color="blue")
    ax.hist(no_hospitals, bins=20, alpha=0.6, label="No", color="orange")
    ax.set_title("Price per Sq Ft Distribution by Nearby Hospitals")
    ax.set_xlabel("Price per Sq Ft")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    

    # 14. Price by Furnished Status
    st.subheader("14. Price by Furnished Status")
    fig, ax = plt.subplots(figsize=(6,4))
    furnished = df[df["furnished_status"]=="Furnished"]["price_in_lakhs"].dropna()
    semi_furnished = df[df["furnished_status"]=="Semi-Furnished"]["price_in_lakhs"].dropna()
    unfurnished = df[df["furnished_status"]=="Unfurnished"]["price_in_lakhs"].dropna()
    ax.hist(furnished, bins=20, alpha=0.6, label="Furnished", color="purple")
    ax.hist(semi_furnished, bins=20, alpha=0.6, label="Semi-Furnished", color="cyan")
    ax.hist(unfurnished, bins=20, alpha=0.6, label="Unfurnished", color="gray")
    ax.set_title("Price Distribution by Furnished Status")
    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # 15. Price per Sq Ft by Facing Direction
    st.subheader("15. Price per Sq Ft by Facing Direction")
    facing_price = df.groupby("facing")["price_per_sqft"].mean().dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(facing_price.index, facing_price.values, color="steelblue")
    ax.set_title("Average Price per Sq Ft by Facing Direction")
    ax.set_xlabel("Facing Direction")
    ax.set_ylabel("Average Price per Sq Ft")
    ax.set_xticklabels(facing_price.index, rotation=45)
    st.pyplot(fig)

    # 16. Number of Properties by Owner Type
    st.subheader("16. Number of Properties by Owner Type")
    owner_counts = df["owner_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(owner_counts.index, owner_counts.values, color="teal")
    ax.set_title("Properties by Owner Type")
    ax.set_xlabel("Owner Type")
    ax.set_ylabel("Number of Properties")
    ax.set_xticklabels(owner_counts.index, rotation=45)
    st.pyplot(fig)

    # 17. Properties by Availability Status
    st.subheader("17. Properties by Availability Status")
    availability_counts = df["availability_status"].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(availability_counts.index, availability_counts.values, color="orange")
    ax.set_title("Properties by Availability Status")
    ax.set_xlabel("Availability Status")
    ax.set_ylabel("Number of Properties")
    ax.set_xticklabels(availability_counts.index, rotation=45)
    st.pyplot(fig)

    # 18. Parking Space vs Price
    st.subheader("18. Parking Space vs Price")
    fig, ax = plt.subplots(figsize=(6,4))
    yes_parking = df[df["parking_space"]=="Yes"]["price_in_lakhs"].dropna()
    no_parking = df[df["parking_space"]=="No"]["price_in_lakhs"].dropna()
    ax.hist(yes_parking, bins=20, alpha=0.6, label="Yes", color="green")
    ax.hist(no_parking, bins=20, alpha=0.6, label="No", color="red")
    ax.set_title("Price Distribution by Parking Space")
    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # 19. Amenities vs Price per Sq Ft
    st.subheader("19. Amenities Count vs Price per Sq Ft")

    amenities_price = df.groupby("amenities_count")["price_per_sqft"].mean()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(amenities_price.index, amenities_price.values, color="steelblue")

    ax.set_xlabel("Number of Amenities")
    ax.set_ylabel("Average Price per Sq Ft")
    ax.set_title("Amenities vs Price per Sq Ft")
    st.pyplot(fig)


    # 20. Public Transport Accessibility vs Price per Sq Ft
    st.subheader("20. Public Transport Accessibility vs Price per Sq Ft")
    fig, ax = plt.subplots(figsize=(6,4))
    high_pt = df[df["public_transport_accessibility"]=="High"]["price_per_sqft"].dropna()
    medium_pt = df[df["public_transport_accessibility"]=="Medium"]["price_per_sqft"].dropna()
    low_pt = df[df["public_transport_accessibility"]=="Low"]["price_per_sqft"].dropna()
    ax.hist(high_pt, bins=20, alpha=0.6, label="High", color="blue")
    ax.hist(medium_pt, bins=20, alpha=0.6, label="Medium", color="orange")
    ax.hist(low_pt, bins=20, alpha=0.6, label="Low", color="gray")
    ax.set_title("Price per Sq Ft Distribution by Public Transport Accessibility")
    ax.set_xlabel("Price per Sq Ft")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    

elif menu == "Prediction":
    
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", df["city"].unique())
        bhk = st.slider("BHK", 1, 6, 2)
        size = st.number_input("Size (sqft)", 300, 5000, 1000)
        price = st.number_input("Current Price (Lakhs)", 10, 1000, 60)

    with col2:
        age = st.slider("Property Age (Years)", 0, 30, 5)
        schools = st.slider("Nearby Schools", 0, 10, 2)
        amenities = st.slider("Amenities Count", 0, 10, 4)

    price_per_sqft = (price * 100000) / size

    input_data = {
        "bhk": bhk,
        "size_in_sqft": size,
        "price_in_lakhs": price,
        "age_of_property": age,
        "nearby_schools": schools,
        "amenities_count": amenities,
        "price_per_sqft": price_per_sqft
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encode city
    city_dummies = pd.get_dummies(pd.Series([city]), prefix="city")
    input_df = pd.concat([input_df, city_dummies], axis=1)

    # Align columns with model
    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_cols]

    st.markdown("---")

    if st.button("🔮 Predict Investment"):
        invest = clf.predict(input_df)[0]
        future_price = reg.predict(input_df)[0]

        st.metric("Estimated Price After 5 Years", f"₹ {future_price:.2f} Lakhs")

        if invest == 1:
            st.success("✅ GOOD Investment")
        else:
            st.error("❌ NOT a Good Investment")
