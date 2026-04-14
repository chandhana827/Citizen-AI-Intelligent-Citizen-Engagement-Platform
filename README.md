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

I built this to explore what AI-assisted civic engagement could look like — a single interface where citizens can ask questions, report issues, analyze sentiment, and get responses tailored to where they live and what they care about.

It runs on IBM's Granite 3B model and is built entirely with Gradio. You can run it on Google Colab for free whenever you need it, or deploy it permanently on Hugging Face Spaces if you want it always online.

---

## What it does

**Chat assistant** — ask anything civic-related and Granite 3B responds. Local government schemes, how to file a complaint, what a policy means — it handles most of it reasonably well.

**Sentiment analysis** — paste any citizen comment and get an instant positive/negative reading with a confidence score. Good for quickly gauging public mood on a topic.

**Live feedback dashboard** — citizens submit feedback under a category (healthcare, transport, education, sanitation) and a live bar chart updates showing the sentiment breakdown across categories.

**Personalized responses** — enter a user ID and the AI responds with awareness of your city and known local issues. Currently has two demo profiles (Hyderabad and Delhi) but easy to extend.

---

## Quickest way to run it — Google Colab

If you don't have a GPU at home and don't want to deal with deployment, Google Colab is the most practical option. It gives you a free T4 GPU, takes about 5 minutes to set up, and generates a public link you can share with anyone.

The link lasts as long as your Colab session is active (usually up to 12 hours on the free tier). When you need it again, just run the notebook again and you get a fresh link.

### Step 1 — Open a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

Before running anything, switch to a GPU runtime:
**Runtime → Change runtime type → T4 GPU → Save**

### Step 2 — Install dependencies

Paste this into the first cell and run it:

```python
!pip install gradio==4.44.0
!pip install transformers==4.44.0
!pip install torch==2.3.1
!pip install accelerate==0.33.0
!pip install sentencepiece==0.2.0
!pip install matplotlib==3.9.1
!pip install pandas==2.2.2
!pip install scipy==1.13.1
```

This takes 3–5 minutes. Run it once per session.

### Step 3 — Paste and run the app

Create a new cell, paste the entire contents of `app.py` into it, but change the very last line from:

```python
demo.launch(share=True)
```

to:

```python
demo.launch(share=True, debug=True)
```

Run the cell. After the model loads (takes 2–3 minutes the first time), you'll see something like this in the output:

```
Running on public URL: https://xxxxxxxxxxxxxx.gradio.live
```

That's your link. Anyone with it can use the app.

### Step 4 — Keep the session alive

Colab disconnects if it thinks you're idle. To prevent this, open the browser console on the Colab page (F12 → Console) and paste:

```javascript
function KeepAlive() {
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(KeepAlive, 60000)
```

This clicks the reconnect button every 60 seconds and keeps your session from timing out.

### What to expect

- First run: model downloads ~6 GB, takes 5–10 minutes
- After that: model loads from Colab's cache, takes about 2 minutes
- Session limit: ~12 hours on free Colab, then you need to restart
- The public link changes every session — it's not the same URL each time

---

## Deploying on Hugging Face Spaces (permanent option)

If you want it always online without running Colab manually, Hugging Face Spaces is the way to go. It's free, supports Gradio natively, and gives you a permanent URL.

Go to [huggingface.co/new-space](https://huggingface.co/new-space), name your Space, select Gradio as the SDK, and pick the T4 Small GPU hardware tier — it's free and enough for this app.

Upload three files:

```
app.py
requirements.txt
README.md
```

Before uploading, change the last line of `app.py` to just:

```python
demo.launch()
```

Spaces provides the public URL automatically, so `share=True` isn't needed.

Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/citizen-ai` within about 10 minutes of the first build.

One thing to know — on the free tier, Spaces goes to sleep after a period of inactivity and takes about 2 minutes to wake up when someone visits. It's not truly always-on, but it's permanent and doesn't require you to do anything to keep it alive.

If you want to connect your GitHub repo so the Space redeploys automatically whenever you push changes, go to Space Settings → Repository and link it there.

---

## Running locally

If you have your own NVIDIA GPU with at least 4 GB of VRAM, running locally is straightforward:

```bash
git clone https://github.com/YOUR_USERNAME/citizen-ai.git
cd citizen-ai
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

First run downloads the model (~6 GB). After that it's cached. The terminal will print a local URL and, because `share=True` is set, also a public Gradio link valid for 72 hours.

---

## requirements.txt

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

---

## Things worth knowing

**Feedback data resets on restart.** The dashboard stores submissions in a Python list in memory. Every time the app restarts — whether on Colab, Spaces, or locally — that data is gone. For a real deployment you'd want to write it to a file or database.

**User profiles are hardcoded.** There are two demo profiles in the script (user IDs 1001 and 1002). Adding more is just adding entries to the `user_profiles` dictionary in `app.py` — no database needed for demo purposes.

**Out of memory errors.** If the app crashes with a CUDA OOM error, open `app.py` and lower `max_new_tokens` from 200 to 100 in both `model.generate()` calls. That usually fixes it.

**Colab Pro vs free.** Colab Pro ($10/month) gives you longer sessions (up to 24 hours), faster GPUs, and more RAM. If you're using this frequently, it's worth it. But the free tier is totally fine for demos and occasional use.

---

## Swapping the model

Granite 3B is good but heavy. If you want something that loads faster or uses less memory, change `model_id` at the top of `app.py`:

```python
model_id = "microsoft/phi-2"             # smaller and faster, quality holds up well
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # very light, works on free Colab easily
model_id = "mistralai/Mistral-7B-v0.1"   # better quality but needs more VRAM
```

Any instruction-tuned causal LM from HuggingFace should work as a drop-in replacement.
---
Demo Video 


https://github.com/user-attachments/assets/a36d0a80-794f-4474-b110-19b4b1e233e1

---

## License

MIT. Use it, modify it, build on it.

