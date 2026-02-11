import os
import sys
import uuid
from urllib.parse import quote_plus

if sys.version_info >= (3, 13):
    import types
    sys.modules["cgi"] = types.ModuleType("cgi")

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

import requests
import feedparser
from flask import Flask, render_template, request, session, jsonify
from openai import OpenAI

# ================= CONFIG =================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

client = OpenAI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Store telegram conversations in memory
telegram_conversations = {}

# =================================================
#                 PAPER FETCHERS
# =================================================

def fetch_arxiv_papers(query, max_results=3):
    url = f"{ARXIV_API}?search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
    feed = feedparser.parse(url)
    return [{"title": e.title, "summary": e.summary, "link": e.link} for e in feed.entries]


def fetch_pubmed_papers(query, max_results=3):
    search = requests.get(PUBMED_API, params={
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }, timeout=10).json()

    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    abstracts = requests.get(PUBMED_FETCH, params={
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "text",
        "rettype": "abstract"
    }, timeout=10).text

    return [{
        "title": f"PubMed Article {pid}",
        "summary": abstracts[:1000],
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
    } for pid in ids]

# =================================================
#                 AI ANSWER WITH MEMORY
# =================================================

def answer_with_memory(chat_messages, question):

    papers = fetch_arxiv_papers(question) + fetch_pubmed_papers(question)

    research_context = "\n\n".join(
        f"Title: {p['title']}\nSummary: {p['summary']}\nSource: {p['link']}"
        for p in papers
    )

    system_prompt = {
        "role": "system",
        "content": (
            "You are a medical research assistant.\n"
            "Use previous conversation context.\n"
            "Ground answers in scientific research.\n"
            "If evidence is weak, say so clearly."
        )
    }

    messages = [system_prompt]
    messages.extend(chat_messages[-10:])

    messages.append({
        "role": "user",
        "content": f"RESEARCH:\n{research_context}\n\nQUESTION:\n{question}"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()

# =================================================
#                     TELEGRAM
# =================================================

def send_telegram_message(chat_id, text):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text}
    )

@app.route("/telegram_webhook", methods=["POST"])
def telegram_webhook():
    data = request.json

    if "message" not in data:
        return jsonify({"status": "ignored"})

    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"].get("text", "")

    if not user_text:
        return jsonify({"status": "no text"})

    # Create conversation memory per Telegram user
    if chat_id not in telegram_conversations:
        telegram_conversations[chat_id] = []

    chat_history = telegram_conversations[chat_id]

    chat_history.append({"role": "user", "content": user_text})

    answer = answer_with_memory(chat_history, user_text)

    chat_history.append({"role": "assistant", "content": answer})

    send_telegram_message(chat_id, answer)

    return jsonify({"status": "ok"})

# =================================================
#                     WEB ROUTES
# =================================================

@app.route("/")
def home():
    session.setdefault("conversations", {})
    session.setdefault("active_chat", None)

    if session["active_chat"] is None:
        chat_id = str(uuid.uuid4())
        session["conversations"][chat_id] = []
        session["active_chat"] = chat_id

    return render_template(
        "index.html",
        conversations=session["conversations"],
        active_chat=session["active_chat"]
    )

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    user_input = data.get("message", "").strip()

    chat_id = session["active_chat"]
    chat_messages = session["conversations"][chat_id]

    chat_messages.append({"role": "user", "content": user_input})
    answer = answer_with_memory(chat_messages, user_input)
    chat_messages.append({"role": "assistant", "content": answer})

    session.modified = True

    return jsonify({"bot": answer})

# =================================================
#                     MAIN
# =================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
