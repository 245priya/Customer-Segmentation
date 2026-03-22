import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Customer Segmentation + Recommendation",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation + Recommendation System")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Data Preview")
    st.write(df.head())

    # ---------------- FEATURE SELECTION ----------------
    st.sidebar.header("⚙️ Settings")

    features = st.sidebar.multiselect(
        "Select Features",
        options=df.columns,
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    if len(features) < 2:
        st.warning("Please select at least 2 features")
        st.stop()

    df = df[features]

    # ---------------- PREPROCESSING ----------------
    df = df.dropna()

    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = encoder.fit_transform(df[col])

    # ---------------- ELBOW METHOD ----------------
    st.subheader("📉 Elbow Method")

    inertia = []
    k_range = range(1, 11)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(df)
        inertia.append(model.inertia_)

    elbow_df = pd.DataFrame({"K": k_range, "Inertia": inertia})
    st.line_chart(elbow_df.set_index("K"))

    # Find optimal K
    KL = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
    optimal_k = KL.elbow

    if optimal_k is None:
        optimal_k = 3

    st.success(f"✅ Optimal number of clusters: {optimal_k}")

    # ---------------- MODEL TRAINING ----------------
    model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(df)

    # ---------------- VISUALIZATION ----------------
    st.subheader("📍 Cluster Visualization")

    st.scatter_chart(
        df,
        x=features[0],
        y=features[1],
        color="Cluster"
    )

    # ---------------- CLUSTER INSIGHTS ----------------
    st.subheader("📊 Cluster Insights")
    insights = df.groupby("Cluster").mean()
    st.dataframe(insights)

    # ---------------- RECOMMENDATION SYSTEM ----------------
    st.subheader("💡 Customer Recommendations")

    for i in range(optimal_k):
        st.markdown(f"### 🔹 Cluster {i}")

        avg_income = insights.iloc[i][0]
        avg_score = insights.iloc[i][1]

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Average Income: {round(avg_income,2)}")
            st.write(f"Average Spending Score: {round(avg_score,2)}")

        with col2:
            if avg_income > 50 and avg_score > 50:
                st.success("💎 Premium Customers → Offer luxury products, VIP membership, loyalty rewards")

            elif avg_income > 50 and avg_score <= 50:
                st.warning("🎯 Target Customers → Give discounts and personalized offers to increase spending")

            elif avg_income <= 50 and avg_score > 50:
                st.info("⚠️ Impulsive Buyers → Promote budget-friendly premium products")

            else:
                st.error("🛒 Low Value Customers → Minimal marketing, basic offers only")

else:
    st.info("👆 Please upload a CSV file to begin")