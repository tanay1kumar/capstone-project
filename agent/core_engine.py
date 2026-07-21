# core engine - generic browser agent, no site-specific logic in here
# all navigation instructions come from the skill's SKILL.md file
# to add a new site just drop a new skills/<name>/SKILL.md, no code changes needed

import asyncio
import logging
import shutil
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from browser_use.llm import ChatAnthropic
from langgraph.graph import StateGraph, END
from browser_use import Agent as BrowserAgent
from browser_use.browser.profile import BrowserProfile

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MAX_RETRIES = 3


class AgentState(TypedDict):
    skill_name: str
    skill_instructions: str
    download_dir: str
    rag_data_dir: str
    status: str
    error: str
    attempts: int


def load_skill(state: AgentState) -> AgentState:
    log.info(f"Loading skill: {state['skill_name']}")

    skill_path = Path("skills") / state["skill_name"] / "SKILL.md"
    if not skill_path.exists():
        return {**state, "status": "failed", "error": f"SKILL.md not found at {skill_path}"}

    instructions = skill_path.read_text()
    download_dir = Path("skills") / state["skill_name"] / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    return {
        **state,
        "skill_instructions": instructions,
        "download_dir": str(download_dir),
        "status": "pending",
    }


async def run_browser(state: AgentState) -> AgentState:
    attempt = state["attempts"] + 1
    log.info(f"Browser agent starting (attempt {attempt}/{MAX_RETRIES})")

    try:
        llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

        browser_profile = BrowserProfile(
            headless=True,
            accept_downloads=True,
            downloads_path=state["download_dir"],
        )

        # the SKILL.md content is passed directly as the task
        # the LLM reads it and figures out what to click/navigate
        agent = BrowserAgent(
            task=state["skill_instructions"],
            llm=llm,
            browser_profile=browser_profile,
        )

        await agent.run()

        return {**state, "attempts": attempt, "status": "checking"}

    except Exception as e:
        log.error(f"Browser agent error: {e}")
        return {**state, "attempts": attempt, "status": "failed", "error": str(e)}


def check_result(state: AgentState) -> AgentState:
    download_dir = Path(state["download_dir"])
    pdfs = list(download_dir.glob("*.pdf"))

    if not pdfs:
        log.warning(f"No PDFs found in {download_dir}")
        return {**state, "status": "failed", "error": "No PDFs downloaded"}

    log.info(f"Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")

    # copy downloads to the RAG data folder
    rag_dir = Path(state["rag_data_dir"])
    rag_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        dest = rag_dir / pdf.name
        shutil.copy2(pdf, dest)
        log.info(f"Copied {pdf.name} -> {dest}")

    return {**state, "status": "success"}


def should_retry(state: AgentState) -> str:
    if state["status"] == "success":
        return END

    if state["attempts"] < MAX_RETRIES:
        log.warning(f"Retrying... ({state['attempts']}/{MAX_RETRIES} attempts used)")
        return "run_browser"

    log.error(f"Max retries reached. Last error: {state['error']}")
    return END


def route_after_load(state: AgentState) -> str:
    if state["status"] == "failed":
        return END
    return "run_browser"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_skill", load_skill)
    graph.add_node("run_browser", run_browser)
    graph.add_node("check_result", check_result)

    graph.set_entry_point("load_skill")
    graph.add_conditional_edges("load_skill", route_after_load)
    graph.add_edge("run_browser", "check_result")
    graph.add_conditional_edges("check_result", should_retry)

    return graph.compile()


async def run_skill(skill_name: str, rag_data_dir: str = "data/sec_regulations") -> dict:
    graph = build_graph()

    initial_state: AgentState = {
        "skill_name": skill_name,
        "skill_instructions": "",
        "download_dir": "",
        "rag_data_dir": rag_data_dir,
        "status": "pending",
        "error": "",
        "attempts": 0,
    }

    return await graph.ainvoke(initial_state)
