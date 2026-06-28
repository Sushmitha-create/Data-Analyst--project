# IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Traffic Accident Dashboard",
    layout="wide"
)

# TITLE
st.markdown("""
<h1 style='text-align:center; color:red;'>
🚗 Traffic Accident Analysis Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# LOAD DATASET
@st.cache_data
def load_data():

    df = pd.read_csv(
        "traffic_crashes.csv"
    )

    return df

df = load_data()


# SELECT IMPORTANT COLUMNS
df = df[[

    'WEATHER_CONDITION',

    'LIGHTING_CONDITION',

    'ROAD_DEFECT',

    'FIRST_CRASH_TYPE',

    'PRIM_CONTRIBUTORY_CAUSE',

    'CRASH_HOUR',

    'CRASH_DAY_OF_WEEK',

    'CRASH_MONTH',

    'INJURIES_TOTAL'
]]

# CLEAN DATA
df = df.dropna()
# CREATE SEVERITY COLUMN
def severity(x):

    if x == 0:
        return "Low"

    elif x <= 2:
        return "Medium"

    else:
        return "High"

df['SEVERITY'] = df[
    'INJURIES_TOTAL'
].apply(severity)

# SIDEBAR FILTERS
st.sidebar.header("🔍 Dashboard Filters")
# Weather Filter
weather_filter = st.sidebar.multiselect(

    "Select Weather Condition",

    options=df['WEATHER_CONDITION'].unique(),

    default=df['WEATHER_CONDITION'].unique()
)
# Severity Filter
severity_filter = st.sidebar.multiselect(

    "Select Severity",

    options=df['SEVERITY'].unique(),

    default=df['SEVERITY'].unique()
)
# Lighting Filter
lighting_filter = st.sidebar.multiselect(

    "Select Lighting Condition",

    options=df['LIGHTING_CONDITION'].unique(),

    default=df['LIGHTING_CONDITION'].unique()
)
# Month Filter
month_filter = st.sidebar.slider(

    "Select Crash Month",

    1,

    12,

    (1,12)
)

# Hour Filter
hour_filter = st.sidebar.slider(

    "Select Crash Hour",

    0,

    23,

    (0,23)
)

# APPLY FILTERS
filtered_df = df[

    (df['WEATHER_CONDITION'].isin(weather_filter)) &

    (df['SEVERITY'].isin(severity_filter)) &

    (df['LIGHTING_CONDITION'].isin(lighting_filter)) &

    (df['CRASH_MONTH'] >= month_filter[0]) &

    (df['CRASH_MONTH'] <= month_filter[1]) &

    (df['CRASH_HOUR'] >= hour_filter[0]) &

    (df['CRASH_HOUR'] <= hour_filter[1])
]

# KPI METRICS
st.subheader("📊 Dashboard Overview")

total_accidents = len(filtered_df)

total_injuries = int(
    filtered_df['INJURIES_TOTAL'].sum()
)

high_cases = len(

    filtered_df[
        filtered_df['SEVERITY'] == 'High'
    ]
)

avg_injuries = round(

    filtered_df['INJURIES_TOTAL'].mean(),

    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🚘 Total Accidents",
    total_accidents
)

col2.metric(
    "🏥 Total Injuries",
    total_injuries
)

col3.metric(
    "⚠ High Severity",
    high_cases
)

col4.metric(
    "📈 Avg Injuries",
    avg_injuries
)

st.markdown("---")

# WEATHER ANALYSIS
col1, col2 = st.columns(2)

with col1:

    st.subheader("🌦 Weather Analysis")

    weather_counts = filtered_df[
        'WEATHER_CONDITION'
    ].value_counts().head(5)

    fig1, ax1 = plt.subplots(
        figsize=(8,5)
    )

    sns.barplot(

        x=weather_counts.index,

        y=weather_counts.values,

        palette='viridis',

        ax=ax1
    )

    plt.xticks(rotation=20)

    for i, value in enumerate(weather_counts.values):

        ax1.text(

            i,

            value + 100,

            str(value),

            ha='center',

            fontsize=10,

            fontweight='bold'
        )

    plt.title(
        "Top Weather Conditions",
        fontsize=16,
        fontweight='bold'
    )

    st.pyplot(fig1)

# SEVERITY PIE CHART
with col2:

    st.subheader("🚨 Severity Distribution")

    severity_counts = filtered_df[
        'SEVERITY'
    ].value_counts()

    colors = [
        '#4CAF50',
        '#FF9800',
        '#F44336'
    ]

    fig2, ax2 = plt.subplots(
        figsize=(7,7)
    )

    ax2.pie(

        severity_counts.values,

        labels=severity_counts.index,

        autopct='%1.1f%%',

        startangle=140,

        colors=colors,

        shadow=True,

        explode=(0.03, 0.03, 0.08),

        textprops={
            'fontsize':12,
            'fontweight':'bold'
        }
    )

    plt.title(
        "Severity Distribution",
        fontsize=16,
        fontweight='bold'
    )

    st.pyplot(fig2)

# LIGHTING CONDITION ANALYSIS

st.markdown("---")

st.subheader("💡 Lighting Condition Analysis")

lighting_counts = filtered_df[
    'LIGHTING_CONDITION'
].value_counts().head(10)

fig3, ax3 = plt.subplots(
    figsize=(12,6)
)

sns.barplot(

    x=lighting_counts.index,

    y=lighting_counts.values,

    palette='magma',

    ax=ax3
)

ax3.set_title(

    "Top Lighting Conditions Causing Accidents",

    fontsize=18,

    fontweight='bold'
)

ax3.set_xlabel(
    "Lighting Condition",
    fontsize=14
)

ax3.set_ylabel(
    "Number of Accidents",
    fontsize=14
)

plt.xticks(
    rotation=25,
    ha='right'
)

for i, value in enumerate(lighting_counts.values):

    ax3.text(

        i,

        value + 100,

        str(value),

        ha='center',

        fontsize=10,

        fontweight='bold'
    )

ax3.grid(
    axis='y',
    linestyle='--',
    alpha=0.3
)

plt.tight_layout()

st.pyplot(fig3)

# LIGHTING DISTRIBUTION PIE CHART

st.markdown("---")

st.subheader("💡 Lighting Condition Distribution")

lighting_distribution = filtered_df[
    'LIGHTING_CONDITION'
].value_counts().head(5)

fig4, ax4 = plt.subplots(
    figsize=(7,7)
)

colors = sns.color_palette(
    'magma',
    len(lighting_distribution)
)

ax4.pie(

    lighting_distribution.values,

    labels=lighting_distribution.index,

    autopct='%1.1f%%',

    startangle=140,

    colors=colors,

    shadow=True,

    textprops={
        'fontsize':11,
        'fontweight':'bold'
    }
)

plt.title(

    "Lighting Condition Distribution",

    fontsize=18,

    fontweight='bold'
)

st.pyplot(fig4)

# CRASH TYPE VS SEVERITY

st.markdown("---")

st.subheader(
    "🚗 First Crash Type vs Severity Analysis"
)

pivot_table = pd.crosstab(

    filtered_df['FIRST_CRASH_TYPE'],

    filtered_df['SEVERITY']
)

top_crashes = pivot_table.sum(axis=1).sort_values(

    ascending=False
).head(10).index

pivot_table = pivot_table.loc[top_crashes]

fig5, ax5 = plt.subplots(
    figsize=(15,8)
)

pivot_table.plot(

    kind='bar',

    stacked=True,

    colormap='Set2',

    edgecolor='black',

    ax=ax5
)

plt.title(

    "First Crash Type vs Severity Analysis",

    fontsize=20,

    fontweight='bold'
)

plt.xlabel(
    "Crash Type",
    fontsize=14
)

plt.ylabel(
    "Accident Count",
    fontsize=14
)

plt.xticks(
    rotation=30,
    ha='right'
)

plt.legend(

    title='Severity',

    bbox_to_anchor=(1.02,1),

    loc='upper left'
)

totals = pivot_table.sum(axis=1)

for i, total in enumerate(totals):

    ax5.text(

        i,

        total + 100,

        str(int(total)),

        ha='center',

        fontsize=10,

        fontweight='bold'
    )

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.3
)

plt.tight_layout()

st.pyplot(fig5)

# MONTHLY TREND
st.markdown("---")

col3, col4 = st.columns(2)

with col3:

    st.subheader("📈 Monthly Accident Trend")

    monthly_counts = filtered_df[
        'CRASH_MONTH'
    ].value_counts().sort_index()

    fig6, ax6 = plt.subplots(
        figsize=(8,5)
    )

    sns.lineplot(

        x=monthly_counts.index,

        y=monthly_counts.values,

        marker='o',

        linewidth=3,

        ax=ax6
    )

    plt.xlabel("Month")

    plt.ylabel("Accidents")

    plt.grid(alpha=0.3)

    st.pyplot(fig6)

# HOURLY TREND
with col4:

    st.subheader("🕒 Hourly Accident Trend")

    hour_counts = filtered_df[
        'CRASH_HOUR'
    ].value_counts().sort_index()

    fig7, ax7 = plt.subplots(
        figsize=(8,5)
    )

    sns.lineplot(

        x=hour_counts.index,

        y=hour_counts.values,

        marker='o',

        linewidth=3,

        color='red',

        ax=ax7
    )

    plt.xlabel("Hour")

    plt.ylabel("Accidents")

    plt.grid(alpha=0.3)

    st.pyplot(fig7)

# MACHINE LEARNING MODEL

st.markdown("---")

st.subheader("🤖 Random Forest Classifier")

ml_df = filtered_df.copy()

categorical_columns = [

    'WEATHER_CONDITION',

    'LIGHTING_CONDITION',

    'ROAD_DEFECT',

    'FIRST_CRASH_TYPE',

    'PRIM_CONTRIBUTORY_CAUSE',

    'SEVERITY'
]

le = LabelEncoder()

for col in categorical_columns:

    ml_df[col] = le.fit_transform(
        ml_df[col]
    )

# Features and Target
X = ml_df.drop(

    ['SEVERITY', 'INJURIES_TOTAL'],

    axis=1
)
y = ml_df['SEVERITY']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# Train Model
model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)

model.fit(
    X_train,
    y_train
)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

st.success(
    f"✅ Model Accuracy : {accuracy*100:.2f}%"
)

# CONFUSION MATRIX
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig8, ax8 = plt.subplots(
    figsize=(6,5)
)
sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    ax=ax8
)

plt.title(
    "Confusion Matrix",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

st.pyplot(fig8)

# SHOW FILTERED DATA

st.markdown("---")

st.subheader("📄 Filtered Dataset")

st.dataframe(filtered_df.head(100))

# FOOTER
st.markdown("---")

st.markdown("""
<center>
<h4>
Traffic Accident Analysis and Severity Prediction Using Machine Learning
</h4>
</center>
""", unsafe_allow_html=True)