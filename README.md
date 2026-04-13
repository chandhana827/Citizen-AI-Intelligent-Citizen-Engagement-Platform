---
title: Citizen AI – Intelligent Citizen Engagement Platform
emoji: 🧠
colorFrom: blue
colorTo: teal
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Citizen AI

I built this as a way to explore what AI-assisted civic engagement could look like. The idea is simple — give citizens a smart interface to report issues, ask questions, and get responses that are actually relevant to where they live and what they care about.

It's not perfect, but it works. Here's what's inside and how to get it running.

---

## What it does

There are four parts to the app:

**Chat assistant** — a general-purpose chatbot backed by IBM's Granite 3B model. You can ask it anything civic-related: local laws, how to file a complaint, what a government scheme means, whatever. It's not always right, but it handles most questions reasonably well.

**Sentiment analysis** — paste in any citizen comment and it'll tell you whether the tone is positive or negative. Useful if you're processing a batch of feedback and want a quick read on how people are feeling about a particular topic.

**Live feedback dashboard** — citizens submit feedback under a category (healthcare, transport, education, etc.) and the chart updates in real time showing the sentiment breakdown. It's a simple demo of what a live civic feedback tracker could look like.

**Personalized responses** — this one's experimental. You enter a user ID, and the AI responds to your query with awareness of your city and your known issues. Right now it's just two hardcoded profiles (Hyderabad and Delhi), but the idea scales.

---

## Running it locally

You'll need Python 3.10 or newer and a CUDA GPU with at least 4GB of VRAM. The Granite 3B model runs in float16, so it fits on a T4 comfortably, but it won't run well on CPU.

```bash
git clone https://github.com/YOUR_USERNAME/citizen-ai.git
cd citizen-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

First run will download the model weights (~6 GB), so give it some time. After that it caches and subsequent starts are fast. Once it's up, you'll see a local URL in the terminal — open that in your browser.

The app already has `share=True` in the launch call, so you'll also get a public Gradio link that's valid for 72 hours. Good for quick demos without any deployment setup.

---

## Deploying on Hugging Face Spaces

This is the easiest way to keep it online permanently and for free.

Go to [huggingface.co/new-space](https://huggingface.co/new-space), pick Gradio as the SDK, and make sure you select a GPU hardware tier — T4 Small is free and works fine. CPU-only will basically not work with this model.

You only need three files in the Space:

```
app.py
requirements.txt
README.md
```

Upload them manually through the Files tab, or connect your GitHub repo under Space Settings → Repository and it'll auto-deploy on every push. The second option is much nicer once you're past the initial setup.

Before deploying, change the last line of `app.py` from:
```python
demo.launch(share=True)
```
to just:
```python
demo.launch()
```
Spaces gives you a public URL automatically, so `share=True` isn't needed there.

First build takes around 10 minutes because it has to download the model. After that, the Space stays warm and loads quickly.

---

## Dependencies

```
gradio==4.44.0
transformers==4.44.0
torch==2.3.1
accelerate==0.33.0
sentencepiece==0.2.0
matplotlib==3.9.1
pandas==2.2.2
scipy==1.13.1
```

Pin the versions. HF Spaces rebuilds periodically and unpinned dependencies have a way of quietly breaking things.

---

## A few things worth knowing

The feedback dashboard stores data in memory, so it resets every time the app restarts. If you want persistent storage, you'd need to swap that list out for a proper database or even just a CSV write.

The user profiles are hardcoded as a dictionary in the script. Adding new ones is just adding another entry — no database needed for a small demo. If this were a real product you'd want to pull profiles from an actual user table, but for prototyping this is fine.

If you get an out-of-memory error on Spaces, the quickest fix is to lower `max_new_tokens` from 200 to 100 in the `model.generate()` calls. You can also try upgrading to an A10G tier if you have access.

The sentiment pipeline downloads distilbert automatically on first run. It's only about 250MB so it won't cause any issues.

---

## Swapping the model

If Granite 3B feels too heavy or you want to try something different, just change the `model_id` at the top of `app.py`:

```python
model_id = "microsoft/phi-2"            # lighter, still pretty capable
model_id = "mistralai/Mistral-7B-v0.1"  # heavier but noticeably better
```

Any causal LM on HuggingFace that supports `apply_chat_template` should work as a drop-in.

---

## License
Use it, modify it, build on it.

