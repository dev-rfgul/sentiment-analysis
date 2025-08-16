from transformers import pipeline
import pandas as pd
import gradio as gr

# Load Hugging Face sentiment model
analyzer = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Function for a single review
def sentiment_analysis(text: str) -> str:
    sentiment = analyzer(text)[0]
    return sentiment['label']

# Process reviews (string text or Excel file)
def process_reviews(input_text, input_file):
    reviews = []

    # Case 1: Text input
    if input_text:
        reviews.extend([r.strip() for r in input_text.replace("\n", ".").split(".") if r.strip()])

    # Case 2: Excel file
    if input_file:
        df = pd.read_excel(input_file)
        reviews.extend(df.iloc[:, 0].dropna().astype(str).tolist())  # ensure strings

    if not reviews:
        return pd.DataFrame([{"Review": "No input provided", "Sentiment": "N/A"}])

    # Analyze reviews safely
    results = []
    for review in reviews:
        if not isinstance(review, str) or not review.strip():
            results.append({"Review": str(review), "Sentiment": "Invalid/Empty"})
            continue
        try:
            sentiment = sentiment_analysis(review)
            results.append({"Review": review, "Sentiment": sentiment})
        except Exception as e:
            results.append({"Review": review, "Sentiment": f"Error: {e}"})

    # Save to Excel
    df_out = pd.DataFrame(results)
    df_out.to_excel("review_sentiments.xlsx", index=False)

    return df_out

# ---- Gradio UI ----
demo = gr.Interface(
    fn=process_reviews,
    inputs=[
        gr.Textbox(lines=5, placeholder="Paste reviews here (separated by newlines or periods)...", label="Text Reviews"),
        gr.File(file_types=[".xlsx"], type="filepath", label="Upload Excel File")
    ],
    outputs=gr.Dataframe(label="Sentiment Results"),
    title="Sentiment Analysis for Reviews",
    description="Upload an Excel file with reviews (first column) or paste text reviews. Results will also be saved to 'review_sentiments.xlsx'."
)

if __name__ == "__main__":
    demo.launch()
