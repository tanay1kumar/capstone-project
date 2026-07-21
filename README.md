# sec regulations chatbot

rag chatbot for sec regulatory documents + a browser agent that auto downloads the pdfs.

## setup
```bash
cp .env.example .env  # add ANTHROPIC_API_KEY
docker compose build
```

## usage
```bash
docker compose run agent python -m agent.main sec_gov  # download pdfs
docker compose run app python rag/ingest.py            # index them
docker compose up app                                  # start chatbot at localhost:8501
```
