import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Load model path and classifier outside the main loop for efficiency
model_path = ("Models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/"
              "snapshots/714eb0fa89d2f80546fda750413ed43d93601a13")
classifier = pipeline("text-classification", model=model_path)

def get_default_comments():
    """Fetches comments from session state or returns an empty list"""
    return st.session_state.get('all_comments', [])

def store_comment(username, comment):
    """Stores a new comment with sentiment classification in session state"""
    all_comments = get_default_comments()
    comment_dic = {
        'username': username,
        'comment': comment,
        'classification': classify_comment(comment)
    }
    all_comments.append(comment_dic)
    st.session_state['all_comments'] = all_comments

def classify_comment(comment):
    """Classifies the sentiment of a comment using the loaded model"""
    result = classifier(comment)[0]['label']
    return result.upper()  # Ensure consistent capitalization (POSITIVE/NEGATIVE)

def filter_comments(comments, sort_option):
    """Filters comments based on the selected sorting option"""
    if sort_option == "None":
        return comments
    else:
        return [c for c in comments if c['classification'] == sort_option]

def display_comment(comment):
    """Formats a comment with appropriate HTML and black text color"""
    sentiment = comment['classification']
    background_color = {
        'POSITIVE': '#DFF0D8',  # Light green
        'NEGATIVE': '#F2DEDE',  # Light red
    }.get(sentiment, '#FFFFFF')  # Default white for unknown sentiment
    comment_html = f"""
    <div class="comment-box" style="background-color: {background_color}; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 5px 0px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <div class="comment-header" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <span class="username" style="font-weight: bold; color: #333;">{comment['username']}</span>
            <span class="sentiment" style="color: #555; font-style: italic;">{sentiment}</span>
        </div>
        <div class="comment-text" style="color: #333;">{comment['comment']}</div>
    </div>
    """
    st.write(comment_html, unsafe_allow_html=True)

def main():
    """Main function to organize the Streamlit app logic"""
    st.set_page_config(page_title="Sentiment Analyzer", page_icon=":smiley:")  # Set page title and icon

    # Title
    st.markdown("<h1 style='text-align: center; color: #2A61D3; margin-bottom: 30px;'>Sentiment Analyzer</h1>", unsafe_allow_html=True)

    # Video
    st.video("https://youtu.be/jSgdL1zX4h8?si=PLXncne9ga_gGIWq")

    # Add space between title and video
    st.write("")
    st.write("")

    # Input fields and comments
    username = st.text_input("Enter your username", value="User")
    comment = st.text_area("Enter your comment", value="Amazing video")
    if st.button("Add Comment", key="add_comment"):
        store_comment(username, comment)

    sort_option = st.selectbox("Sort Comments by Sentiment:", ["None", "Positive", "Negative"])

    comments = get_default_comments()
    filtered_comments = filter_comments(comments, sort_option)
    st.subheader("Comments")
    for comment in filtered_comments:
        display_comment(comment)

    # Analysis of overall comments
    st.sidebar.title("Overall Comments Analysis")
    st.sidebar.write(f"Total Comments: {len(comments)}")

    positive_comments = [c for c in comments if c['classification'] == 'POSITIVE']
    negative_comments = [c for c in comments if c['classification'] == 'NEGATIVE']

    total_comments = len(comments)
    positive_percentage = len(positive_comments) / total_comments * 100 if total_comments > 0 else 0
    negative_percentage = len(negative_comments) / total_comments * 100 if total_comments > 0 else 0

    st.sidebar.write(f"Positive Comments: {len(positive_comments)} ({positive_percentage:.2f}%)")
    st.sidebar.write(f"Negative Comments: {len(negative_comments)} ({negative_percentage:.2f}%)")

    # Visualization: Pie chart
    if total_comments > 0:
        labels = ['Positive', 'Negative']
        sizes = [len(positive_comments), len(negative_comments)]
        colors = ['#2ecc71', '#e74c3c']  # Green for positive, red for negative

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.sidebar.write("")
        st.sidebar.subheader("Comments Distribution")
        st.sidebar.pyplot(fig)

if __name__ == '__main__':
    main()
