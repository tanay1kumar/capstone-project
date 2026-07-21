import asyncio
import sys
import logging

from agent.core_engine import run_skill

log = logging.getLogger(__name__)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python -m agent.main <skill_name>")
        print("Example: python -m agent.main sec_gov")
        sys.exit(1)

    skill_name = sys.argv[1]
    print(f"running agent for: {skill_name}")

    result = await run_skill(skill_name)

    if result["status"] == "success":
        print("done, pdfs saved to data/sec_regulations/")
        print("run ingest next: python rag/ingest.py")
    else:
        print(f"something went wrong: {result.get('error', 'unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
