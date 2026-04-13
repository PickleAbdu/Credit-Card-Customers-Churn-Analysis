import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"
D3 = px.colors.qualitative.D3

st.set_page_config(
    page_title="Credit Card Churn Analysis",
    page_icon="💳",
    layout="wide"
)

bank_df = pd.read_csv("BankChurners.csv")
bank_df.drop(columns=[
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
], inplace=True)

attrited = bank_df[bank_df['Attrition_Flag'] == 'Attrited Customer']
existing = bank_df[bank_df['Attrition_Flag'] == 'Existing Customer']

st.sidebar.title("💳 Credit Card Churn")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "1. Introduction",
    "2. Business Questions",
    "3. Exploratory Data Analysis",
    "4. Dashboard & Findings",
    "5. Next Steps & Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** BankChurners.csv")
st.sidebar.markdown(f"**Rows:** {len(bank_df):,}")
st.sidebar.markdown(f"**Columns:** {len(bank_df.columns)}")
churn_rate = round(len(attrited) / len(bank_df) * 100, 1)
st.sidebar.markdown(f"**Churn Rate:** {churn_rate}%")


# ── PAGE 1: INTRODUCTION ──────────────────────────────────────────────────────
if page == "1. Introduction":
    st.title("Credit Card Customer Churn Analysis")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Problem Statement")
        st.markdown("""
        A bank manager is facing an increasing number of customers cancelling their credit card services.
        To prevent further revenue loss, the bank needs to **identify customers who are likely to churn**
        so they can be proactively engaged with better offers and personalized services before they leave.

        This analysis explores the behavioral, demographic, and financial patterns that distinguish
        churned customers from existing ones, with the goal of building a foundation for a predictive churn model.
        """)

        st.subheader("Data Source")
        st.markdown("""
        - **Dataset:** Bank Churners (BankChurners.csv)
        - **Source:** [Kaggle — Credit Card Customers by Sakshi Goyal](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
        - **Size:** 10,127 customers × 21 features (after cleaning)
        - **Target Variable:** `Attrition_Flag` — Existing Customer / Attrited Customer
        """)

    with col2:
        st.subheader("Dataset at a Glance")
        total = len(bank_df)
        churned = len(attrited)
        retained = len(existing)

        st.metric("Total Customers", f"{total:,}")
        st.metric("Churned Customers", f"{churned:,}", delta=f"-{churn_rate}%", delta_color="red")
        st.metric("Retained Customers", f"{retained:,}", delta=f"+{100 - churn_rate}%")

    st.markdown("---")
    st.subheader("Column Definitions")

    col_defs = {
        "CLIENTNUM": "Unique customer identifier",
        "Attrition_Flag": "Target — Existing or Attrited customer",
        "Customer_Age": "Customer age in years",
        "Gender": "Male or Female",
        "Dependent_count": "Number of financial dependents",
        "Education_Level": "Highest education level attained",
        "Marital_Status": "Single, Married, Divorced, or Unknown",
        "Income_Category": "Annual income bracket",
        "Card_Category": "Type of credit card (Blue, Silver, Gold, Platinum)",
        "Months_on_book": "Months the customer has been with the bank",
        "Total_Relationship_Count": "Total number of bank products held",
        "Months_Inactive_12_mon": "Months inactive in the last 12 months",
        "Contacts_Count_12_mon": "Bank contacts in the last 12 months",
        "Credit_Limit": "Maximum credit allowed",
        "Total_Revolving_Bal": "Unpaid balance carried month to month",
        "Avg_Open_To_Buy": "Average remaining credit available",
        "Total_Amt_Chng_Q4_Q1": "Change in transaction amount Q4 vs Q1",
        "Total_Trans_Amt": "Total transaction amount (last 12 months)",
        "Total_Trans_Ct": "Total transaction count (last 12 months)",
        "Total_Ct_Chng_Q4_Q1": "Change in transaction count Q4 vs Q1",
        "Avg_Utilization_Ratio": "Average proportion of credit limit used (0–1)",
    }

    df_defs = pd.DataFrame(list(col_defs.items()), columns=["Column", "Description"])
    st.dataframe(df_defs, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(bank_df.sample(10), use_container_width=True)


# ── PAGE 2: BUSINESS QUESTIONS ────────────────────────────────────────────────
elif page == "2. Business Questions":
    st.title("Business Questions")
    st.markdown("The following questions guide this analysis and help frame actionable insights for the bank.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Customer Demographics")
        st.markdown("""
        - Does gender influence churn rate?
        - Which age group has the highest churn rate?
        - Do customers with more dependents churn more?
        - Does education level affect churn?
        - Does marital status relate to churn?
        - Which income bracket loses the most customers?
        """)

        st.markdown("#### Product & Card Usage")
        st.markdown("""
        - Which card category has the highest churn rate?
        - Do customers with more bank products churn less?
        - Do higher income customers hold premium cards?
        - Is there a relationship between card type and education level?
        """)

        st.markdown("#### Engagement & Activity")
        st.markdown("""
        - Do inactive customers churn more?
        - Does the number of bank contacts correlate with churn?
        - How long do churned customers stay before leaving?
        - Do customers who contact the bank more end up churning anyway?
        """)

    with col2:
        st.markdown("#### Financial Behavior")
        st.markdown("""
        - Do churned customers have lower credit limits?
        - Is there a difference in revolving balance between churned and existing customers?
        - Do churned customers utilize their credit less?
        - Is there a spending threshold below which customers are likely to churn?
        """)

        st.markdown("#### Transaction Behavior")
        st.markdown("""
        - Do churned customers make fewer transactions?
        - Do churned customers spend less in total?
        - Is there a drop in transaction frequency (Q4 vs Q1) among churned customers?
        - What is the typical transaction profile of a churned customer?
        """)

        st.markdown("#### Correlation & Feature Relevance")
        st.markdown("""
        - Are `Credit_Limit` and `Avg_Open_To_Buy` redundant?
        - Which numerical features are most correlated with churn?
        - Are there multicollinear features that should be dropped before modeling?
        """)


# ── PAGE 3: EDA ───────────────────────────────────────────────────────────────
elif page == "3. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    # Target Variable
    st.subheader("Target Variable")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(bank_df, names='Attrition_Flag', title='Customer Attrition Distribution',
                     color_discrete_sequence=D3)
        fig.update_traces(textinfo='label+percent')
        st.plotly_chart(fig)

    with col2:
        fig = px.pie(bank_df, names='Gender', title='Gender Distribution',
                     color_discrete_sequence=D3)
        fig.update_traces(textinfo='label+percent')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Demographics
    st.subheader("Demographics")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x = 'Customer_Age', color = 'Attrition_Flag',
                    title = 'Distribution of Customer Age by Attrition Flag',
                    color_discrete_sequence = px.colors.qualitative.D3,
                    text_auto = '', barmode = 'group',
                    nbins = 20)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(bank_df, x='Marital_Status', color='Attrition_Flag',
                           title='Marital Status by Attrition Flag',
                           color_discrete_sequence=D3,
                            barmode='group', text_auto='')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Income_Category', color='Attrition_Flag',
                           title='Income Category by Attrition Flag',
                           color_discrete_sequence=D3,
                           category_orders = {'Income_Category': ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown']},
                           barmode='group', text_auto='')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(bank_df, x='Education_Level', color='Attrition_Flag',
                           title='Education Level by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Dependent_count', color='Attrition_Flag',
                           title='Dependent Count by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(bank_df, x='Card_Category', color='Attrition_Flag',
                           title='Card Category by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Financial Features
    st.subheader("Financial Behavior")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Credit_Limit', color='Attrition_Flag',
                           title='Credit Limit Distribution by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        

    with col2:
        fig = px.histogram(bank_df, x='Avg_Utilization_Ratio', color='Attrition_Flag',
                           title='Avg Utilization Ratio by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Total_Revolving_Bal', color='Attrition_Flag',
                           title='Total Revolving Balance by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(bank_df, x='Total_Relationship_Count', color='Attrition_Flag',
                           title='Total Relationship Count by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Engagement
    st.subheader("Engagement & Activity")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Months_Inactive_12_mon', color='Attrition_Flag',
                           title='Months Inactive (Last 12 Months) by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(bank_df, x='Months_on_book', color='Attrition_Flag',
                           title='Months on Book by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Transaction Behavior
    st.subheader("Transaction Behavior")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Total_Trans_Ct', color='Attrition_Flag',
                           title='Total Transaction Count by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=20)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(bank_df, x='Total_Ct_Chng_Q4_Q1', color='Attrition_Flag',
                           title='Transaction Count Change Q4 vs Q1 by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=20)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(attrited, x='Education_Level', color='Card_Category',
                           title='Education Level by Card Category (Attrited Customers)',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(bank_df, x='Income_Category', color='Card_Category',
                           title='Income Category vs Card Category',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(bank_df, x='Credit_Limit', y='Avg_Utilization_Ratio',
                 color='Attrition_Flag', title='Credit Limit vs Avg Utilization Ratio',
                 color_discrete_sequence=px.colors.qualitative.D3)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(bank_df, x='Total_Trans_Amt', y='Total_Trans_Ct',
                 color='Attrition_Flag',
                 title='Transaction Amount vs Transaction Count by Attrition Flag',
                 color_discrete_sequence=px.colors.qualitative.D3)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig = px.imshow(bank_df.select_dtypes(include='number').corr(),
                    title='Correlation Heatmap',
                    color_continuous_scale='RdBu_r',
                    text_auto='.2f')
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)


# ── PAGE 4: DASHBOARD & FINDINGS ─────────────────────────────────────────────
elif page == "4. Dashboard & Findings":
    st.title("Final Dashboard & Key Findings")
    st.markdown("A focused summary of the most impactful visualizations and their business implications.")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(bank_df):,}")
    col2.metric("Churn Rate", f"{churn_rate}%")
    col3.metric("Avg Transactions (Churned)", f"{int(attrited['Total_Trans_Ct'].mean())}")
    col4.metric("Avg Transactions (Retained)", f"{int(existing['Total_Trans_Ct'].mean())}")

    st.markdown("---")

    # Finding 1
    st.subheader("1. Transaction Behavior is the Strongest Churn Signal")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(bank_df, x='Total_Trans_Amt', y='Total_Trans_Ct',
                         color='Attrition_Flag',
                         title='Transaction Amount vs Transaction Count',
                         color_discrete_sequence=D3)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(bank_df, x='Total_Trans_Ct', color='Attrition_Flag',
                           title='Transaction Count Distribution',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=20)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Finding:** Churned customers are almost entirely concentrated in the low transaction count and low
    spending zone. Customers who transact fewer than ~40 times per year and spend under $2,000 are at the
    highest churn risk. Transaction behavior is the single strongest predictor of churn in this dataset.
    """)

    st.markdown("---")

    # Finding 2
    st.subheader("2. Low Credit Utilization + Low Credit Limit = High Risk")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(bank_df, x='Credit_Limit', y='Avg_Utilization_Ratio',
                         color='Attrition_Flag',
                         title='Credit Limit vs Avg Utilization Ratio',
                         color_discrete_sequence=D3)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(bank_df, x='Avg_Utilization_Ratio', color='Attrition_Flag',
                           title='Avg Utilization Ratio Distribution',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=20)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""**Finding:** The less the Credit limit, the more the utilization ratio, confirmed by the **-0.48** correlation on the heatmap.
    High limit customers rarely churn.
            A noticable amount of churned customers had 0 balance as shown by the yellow horizontal line where we rarely see any blue dots, that's an indicator that the 0 balance is an indicator that the customer is likely to churn.
            
            Customers with low credit limits and near-zero utilization are the highest churn risk,
            the bank should target this segment proactively with engagement offers or credit limit increases.
    """)

    st.markdown("---")

    # Finding 3
    st.subheader("3. Fewer Bank Relationships = Higher Churn Risk")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Total_Relationship_Count', color='Attrition_Flag',
                           title='Total Relationship Count by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(bank_df, x='Total_Ct_Chng_Q4_Q1', color='Attrition_Flag',
                           title='Transaction Count Change Q4 vs Q1',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group', nbins=10)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Finding:** Customers with only 1–2 bank products churn at a significantly higher rate.
    Deepening the customer relationship through cross-selling is a strong retention lever.
    Additionally, customers with a declining transaction trend (Q4/Q1 ratio < 1.0) are more likely to churn —
    a drop in engagement frequency is an early warning sign.
    """)

    st.markdown("---")

    # Finding 4
    st.subheader("4. Demographic Factors Have Limited Impact")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(bank_df, x='Income_Category', color='Attrition_Flag',
                           title='Income Category by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(bank_df, x='Education_Level', color='Attrition_Flag',
                           title='Education Level by Attrition Flag',
                           color_discrete_sequence=D3,
                           text_auto='', barmode='group')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Finding:** Income category, education level, marital status, and gender show no meaningful
    difference in churn rate. Churn is driven by behavioral patterns — not who the customer is,
    but how they use (or stop using) their card.
    """)

    st.markdown("---")

    # Heatmap
    st.subheader("5. Correlation Summary")
    fig = px.imshow(bank_df.select_dtypes(include='number').corr(),
                    title='Correlation Heatmap',
                    color_continuous_scale='RdBu_r',
                    text_auto='.2f')
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Key correlations:**
*Findings of this HeatMap:*

Positive Corr:
1. Correlation between Months_On_Book and Customer_Age are 0.79 meaning that the older they get, the more they use their credit card.
2. Full correlation between Credit Limit and Avg_open_To_buy which means they're identical, one of them could be dropped.
3. Correlation between Total Transactions and Total Transactions Amount are 0.81 which is Intuitive.
4. Total_Revolving_Bal and Avg_Utilization_Ratio (0.62), higher balance means higher utilization, expected.

Negative Corr:
1. Credit_Limit and Avg_Utilization_Ratio (-0.48), customers with higher credit limits tend to use a smaller proportion of it.
2. Avg_Open_To_Buy and Avg_Utilization_Ratio (-0.54), same logic, more available credit = lower utilization

Notable Corr:
1. Total_Relationship_Count and Total_Trans_Amt (-0.35), customers with more products actually spend less per transaction, interesting.
2. Total_Relationship_Count and Total_Trans_Ct (-0.24), same pattern.
    """)


# ── PAGE 5: NEXT STEPS ────────────────────────────────────────────────────────
elif page == "5. Next Steps & Recommendations":
    st.title("Next Steps & Recommendations")
    st.markdown("---")

    st.subheader("Business Recommendations")
st.markdown("##### Target Low-Activity Customers Early")
st.markdown("""
Customers with fewer than 40 transactions per year and spending under $2,000 should be flagged
as at-risk. The bank should proactively engage them with personalized offers, cashback incentives,
or spending challenges before disengagement leads to churn.
""")

st.markdown("##### Increase Product Cross-Selling")
st.markdown("""
Customers with only 1–2 bank products churn at a significantly higher rate. Offering complementary
products (savings accounts, loans, insurance) deepens the relationship and increases switching costs.
""")
st.markdown("##### Monitor Utilization Drops")
st.markdown("""
A drop in `Avg_Utilization_Ratio` to near zero is a strong early warning signal. Automated alerts
should be triggered when a customer's utilization drops significantly month-over-month.
""")
st.markdown("##### Review Platinum Card Strategy")
st.markdown("""
Platinum card holders show a slightly higher churn rate compared to other categories.
The bank should investigate whether the product benefits are meeting customer expectations
at that tier.
 """)


st.markdown("---")
st.subheader("Summary of Key Findings")

findings = {
        "Finding": [
            "Transaction count & amount",
            "Credit utilization",
            "Bank relationships",
            "Transaction trend (Q4 vs Q1)",
            "Demographics",
            "Credit limit",
        ],
        "Insight": [
            "Strongest churn predictor — low activity customers churn at much higher rates",
            "Near-zero utilization is a strong churn signal, especially with low credit limits",
            "Fewer products = higher churn risk; cross-selling is a key retention lever",
            "Declining transaction frequency (ratio < 1.0) signals upcoming churn",
            "Gender, income, education, marital status show no meaningful churn difference",
            "High-limit customers (>$15k) almost never churn",
        ],
        "Action": [
            "Flag customers with <40 transactions/year for proactive outreach",
            "Trigger alerts when utilization drops to near-zero",
            "Cross-sell complementary products to single-product customers",
            "Monitor Q4/Q1 ratio monthly as an early warning metric",
            "Focus retention budget on behavioral segments, not demographic ones",
            "Consider credit limit increases for at-risk low-limit customers",
        ]
    }

st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
