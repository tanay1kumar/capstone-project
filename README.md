# sec regulations chatbot

rag chatbot for sec regulatory documents + a browser agent that auto downloads the pdfs.

## setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env  # add ANTHROPIC_API_KEY
```

## usage
```bash
python -m agent.main sec_gov   # download pdfs
python rag/ingest.py           # index them
streamlit run app.py           # start chatbot
```
