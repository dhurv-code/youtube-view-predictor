import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("youtube_views_model.pkl")
feature_cols = joblib.load("features.pkl")

st.title("YouTube Video Views Predictor")
st.write("Predict expected views before uploading a video")


CATEGORY_MAP = {
    "Music": 10,
    "Gaming": 20,
    "Education": 27,
    "Science & Technology": 28,
    "Entertainment": 24,
    "News & Politics": 25,
    "Sports": 17,
    "Comedy": 23,
    "People & Blogs": 22
}

title = st.text_input("Video Title")
tags = st.text_input("Tags (separated by |)")

publish_hour = st.slider("Upload Hour (0–23)", 0, 23, 18)

publish_day = st.selectbox(
    "Upload Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)



category_name = st.selectbox(
    "Video Category",
    list(CATEGORY_MAP.keys())
)

category_id = CATEGORY_MAP[category_name]



title_length = len(title)
has_num = int(any(char.isdigit() for char in title))
tags_count = len(tags.split("|")) if tags else 0

day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4,
    "Saturday": 5, "Sunday": 6
}

publish_day_num = day_map[publish_day]


input_data = pd.DataFrame([[
    title_length,
    has_num,
    tags_count,
    publish_hour,
    publish_day_num,
    category_id
]], columns=feature_cols)


if st.button("Predict Views"):
    pred_log = model.predict(input_data)[0]
    predicted_views = np.expm1(pred_log)

    st.success(f"📈 Expected Views: {int(predicted_views):,}")
