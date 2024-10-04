import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_linkedin_job_listings_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to train models
def train_models(targets):
    models = {}
    classification_reports = {}

    for target in targets:
        X = df['job_title']
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)
        models[target] = model
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        classification_reports[target] = report

    return models, classification_reports

# Load the data
df = load_data()

# Sidebar navigation
option = st.sidebar.selectbox("Select Section", ["Sentiment Analysis", "Job Title Prediction"])

if option == "Sentiment Analysis":
    st.title("LinkedIn Job Sentiment Analysis")

    if df is not None:
        df['sentiment_score'] = df['job_summary'].apply(analyze_sentiment)
        df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

        job_title = st.text_input("Enter job title to search:")
        job_titles = df['job_title'].unique().tolist()
        selected_job_title = st.selectbox("Select job title", job_titles)

        if st.button("Search"):
            def search_jobs(job_title):
                results = df[df['job_title'].str.contains(job_title, case=False, na=False)]
                return results[['company_name', 'job_location', 'sentiment', 'sentiment_score', 'job_num_applicants', 'job_posted_time', 'apply_link']]

            search_results = search_jobs(selected_job_title)

            if not search_results.empty:
                st.write(search_results)
            else:
                st.write("No jobs found for the title:", selected_job_title)

    st.header("Visualization")
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Doughnut Chart", "Area Chart"])

    if chart_type == "Bar Chart":
        if not df['sentiment'].value_counts().empty:
            st.bar_chart(df['sentiment'].value_counts())
        else:
            st.error("No data available for Bar Chart")
    elif chart_type == "Pie Chart":
        if not df['sentiment'].value_counts().empty:
            fig, ax = plt.subplots()
            ax.pie(df['sentiment'].value_counts(), labels=df['sentiment'].value_counts().index, autopct='%1.1f%%')
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)
        else:
            st.error("No data available for Pie Chart")
    elif chart_type == "Doughnut Chart":
        if not df['sentiment'].value_counts().empty:
            fig = px.pie(df, values='job_num_applicants', names='sentiment', hole=.3)
            st.plotly_chart(fig)
        else:
            st.error("No data available for Doughnut Chart")
    elif chart_type == "Area Chart":
        if not df['sentiment'].value_counts().empty:
            st.area_chart(df['sentiment'].value_counts())
        else:
            st.error("No data available for Area Chart")

elif option == "Job Title Prediction":
    st.title("Job Title Prediction Dashboard")

    # Ensure required columns exist
    required_columns = [
        'job_title', 'company_name', 'job_location',
        'job_employment_type', 'job_function',
        'job_industries','job_seniority_level'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column: '{col}'.")

    # Define the target variables
    targets = [
        'company_name', 'job_location',
        'job_employment_type', 'job_function',
        'job_industries', 'job_seniority_level'
    ]

    # Train models and get classification reports
    models, classification_reports = train_models(targets)

    # Select job attribute for visualization
    selected_target = st.selectbox("Select Job Attribute for Visualization:", targets, index=0)

    # Create a bar chart for the classification report
    report = classification_reports[selected_target]
    df_report = pd.DataFrame(report).loc[['precision', 'recall', 'f1-score']].T.reset_index()
    df_report.columns = ['class', 'precision', 'recall', 'f1-score']

    fig = px.bar(df_report, x='class', y='f1-score',
                 title=f'Classification Report for {selected_target.replace("_", " ").title()}',
                 labels={'f1-score': 'F1 Score', 'class': 'Classes'})
    st.plotly_chart(fig)

    # Select job title from dropdown
    job_title = st.selectbox("Select Job Title:", df['job_title'].unique())

    # Button to predict job attributes
    if st.button('Predict'):
        if job_title:
            predictions = {}
            for target, model in models.items():
                predictions[target] = model.predict([job_title])[0]

            # Create a DataFrame for displaying predictions
            predictions_df = pd.DataFrame(predictions.items(), columns=['Attribute', 'Predicted Value'])
            st.table(predictions_df)  # Display predictions as a table

            # Create a bar chart of the predictions
            predictions_fig = px.bar(predictions_df, x='Attribute', y='Predicted Value',
                                     title='Predicted Job Attributes',
                                     labels={'Predicted Value': 'Value', 'Attribute': 'Attributes'})
            st.plotly_chart(predictions_fig)  # Display predictions graph
        else:
            st.warning("Please select a job title.")