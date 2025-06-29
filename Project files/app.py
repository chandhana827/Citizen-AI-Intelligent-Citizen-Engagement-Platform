import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Load your model (adjust if needed)
model_id = "ibm-granite/granite-3b-code-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# In-memory storage for feedback
submitted_data = []

# Dummy user profiles
user_profiles = {
    "1001": {"location": "Hyderabad", "issues": ["traffic", "air pollution"]},
    "1002": {"location": "Delhi", "issues": ["waste management", "noise"]},
}

# Chat function
def chat_fn(message, history):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()
    return response

# Sentiment analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return f"{result['label']} ({result['score']*100:.2f}%)"

# Feedback form + live dashboard
def collect_and_plot_feedback(comment, category):
    sentiment = sentiment_analyzer(comment)[0]["label"]
    submitted_data.append({"Category": category, "Sentiment": sentiment})
    
    df = pd.DataFrame(submitted_data)
    summary = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind='bar', stacked=True, ax=ax, colormap="Set2")
    plt.title("Live Citizen Sentiment by Category")
    plt.ylabel("Count")
    plt.tight_layout()
    
    return f"Recorded sentiment: {sentiment}", fig

# Personalized assistant
def personalized_response(user_id, query):
    profile = user_profiles.get(user_id)
    if not profile:
        return "User profile not found. Please check your user ID."
    
    context = f"User from {profile['location']} concerned with: {', '.join(profile['issues'])}. Question: {query}"
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

# Build app
with gr.Blocks(title="Citizen AI – Intelligent Citizen Engagement Platform") as demo:
    gr.Markdown("## 🧠 Citizen AI – Intelligent Citizen Engagement Platform")

    with gr.Tab("🤖 Chat Assistant"):
        chat = gr.ChatInterface(
            fn=chat_fn,
            title="🧠 Ask Citizen AI",
            chatbot=gr.Chatbot(label="Citizen Chat"),
            textbox=gr.Textbox(placeholder="Type your question here...", show_label=False)
        )

    with gr.Tab("📊 Sentiment Analysis"):
        sentiment_input = gr.Textbox(label="Enter citizen comment")
        sentiment_output = gr.Textbox(label="Sentiment Result")
        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(analyze_sentiment, inputs=sentiment_input, outputs=sentiment_output)

    with gr.Tab("📈 Live Dashboard"):
        gr.Markdown("### 📬 Submit Feedback and Watch Sentiment Grow Live")
        comment_input = gr.Textbox(label="Citizen Feedback")
        category_input = gr.Dropdown(choices=["Healthcare", "Sanitation", "Transport", "Education"], label="Category")
        submit_button = gr.Button("Submit Feedback")
        sentiment_display = gr.Textbox(label="Detected Sentiment")
        live_chart = gr.Plot(label="Live Sentiment Chart")
        submit_button.click(collect_and_plot_feedback, inputs=[comment_input, category_input], outputs=[sentiment_display, live_chart])

    with gr.Tab("🧬 Personalized AI Response"):
        uid_input = gr.Textbox(label="User ID (e.g., 1001)")
        query_input = gr.Textbox(label="Your query")
        response_output = gr.Textbox(label="AI Response")
        personal_btn = gr.Button("Generate Personalized Response")
        personal_btn.click(personalized_response, inputs=[uid_input, query_input], outputs=response_output)

demo.launch(share=True)
