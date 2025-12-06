import logging
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.book_recommender.core.logging_config import configure_logging
from src.book_recommender.ml.feedback import get_all_feedback

configure_logging(log_file="analytics.log", log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DeepShelf Analytics Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="collapsed"
)

st.title("DeepShelf Analytics Dashboard")


@st.cache_data
def load_feedback_data():
    """Load and process feedback data."""
    logger.info("Loading feedback data for analytics...")
    feedback_entries = get_all_feedback()
    if not feedback_entries:
        logger.warning("No feedback data found.")
        return pd.DataFrame()

    df = pd.DataFrame(feedback_entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    logger.info(f"Loaded {len(df)} feedback entries.")
    return df


feedback_df = load_feedback_data()

if feedback_df.empty:
    st.info("No feedback data available yet to display analytics.")
else:
    st.header("Key Metrics")
    total_feedback = len(feedback_df)
    positive_feedback = len(feedback_df[feedback_df["feedback"] == "positive"])
    negative_feedback = len(feedback_df[feedback_df["feedback"] == "negative"])
    satisfaction_percentage = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Feedback", total_feedback)
    with col2:
        st.metric("Positive Feedback", positive_feedback)
    with col3:
        st.metric("Negative Feedback", negative_feedback)
    with col4:
        st.metric("Satisfaction Rate", f"{satisfaction_percentage:.1f}%")

    st.markdown("---")

    st.header("Feedback Over Time")
    feedback_by_date = feedback_df.groupby(["date", "feedback"]).size().unstack(fill_value=0)
    fig_time = px.line(
        feedback_by_date,
        x=feedback_by_date.index,
        y=feedback_by_date.columns,
        title="Feedback Count Over Time",
        labels={"value": "Count", "date": "Date", "variable": "Feedback Type"},
        color_discrete_map={"positive": "green", "negative": "red"},
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    st.header("Top Queries")
    top_queries = feedback_df["query"].value_counts().reset_index()
    top_queries.columns = ["Query", "Count"]
    fig_queries = px.bar(
        top_queries.head(10),
        x="Query",
        y="Count",
        title="Top 10 Most Frequent Queries",
        labels={"Count": "Number of Times Queried"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_queries, use_container_width=True)

    st.markdown("---")

    st.header("Most Liked / Disliked Books")

    book_feedback_counts = feedback_df.groupby(["book_title", "feedback"]).size().unstack(fill_value=0)
    book_feedback_counts["net_positive"] = book_feedback_counts.get("positive", 0) - book_feedback_counts.get(
        "negative", 0
    )

    most_liked_books = book_feedback_counts.sort_values(by="net_positive", ascending=False).head(5)
    most_disliked_books = book_feedback_counts.sort_values(by="net_positive", ascending=True).head(5)

    col_liked, col_disliked = st.columns(2)
    with col_liked:
        st.subheader("Top 5 Most Liked Books (Net Positive)")
        if not most_liked_books.empty:
            st.dataframe(
                most_liked_books[["positive", "negative", "net_positive"]].rename(
                    columns={"positive": "👍", "negative": "👎", "net_positive": "Net Score"}
                )
            )
        else:
            st.info("No liked book data.")
    with col_disliked:
        st.subheader("Top 5 Most Disliked Books (Net Negative)")
        if not most_disliked_books.empty:
            st.dataframe(
                most_disliked_books[["positive", "negative", "net_positive"]].rename(
                    columns={"positive": "👍", "negative": "👎", "net_positive": "Net Score"}
                )
            )
        else:
            st.info("No disliked book data.")

    st.markdown("---")

    if st.checkbox("Show Raw Feedback Data"):
        st.subheader("Raw Feedback Data")
        st.dataframe(feedback_df)


def main():
    """Entry point for analytics dashboard"""
    pass


if __name__ == "__main__":
    main()
