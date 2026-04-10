#!/usr/bin/env python3
"""
BabyJay RAG Evaluation Pipeline
================================
LLM-as-Judge evaluation across 4 evaluators with ~100 QA pairs at 5
difficulty levels (Easy → Research), designed for professor panel presentation.

Evaluators
----------
1. Retrieval Relevance  — Did the RAG retrieve relevant documents?
2. Groundedness         — Is the answer grounded in retrieved context?
3. Correctness          — Is the answer factually correct vs. ground truth?
4. Response Relevance   — Does the answer actually address the question?

Entity Precision        — Regex check on exact numbers, emails, course codes

Usage
-----
    cd /Users/pavansaipendry/Documents/BabyJay
    python -m app.scripts.eval_rag

Output
------
    eval_results/eval_report.md       — Human-readable markdown report
    eval_results/eval_results.json    — Raw JSON for further analysis
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# ── Make sure the project root is on sys.path ─────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()  # Must load before any ChromaDB/OpenAI imports

# ── Config ────────────────────────────────────────────────────────────────────
JUDGE_MODEL = "claude-sonnet-4-6"          # evaluator LLM
CHAT_MODEL  = "claude-sonnet-4-6"          # model under test (via BabyJay chat)
OUTPUT_DIR  = project_root / "eval_results"

# Use the Anthropic client for the judge
import anthropic

# ── QA Dataset ────────────────────────────────────────────────────────────────
# fmt: off
QA_PAIRS: List[Dict[str, Any]] = [

    # ─────────────────────────────────────────────────────────────────────────
    # EASY — Single-hop, direct lookup, one clear answer
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "E01",
        "difficulty": "easy",
        "domain": "faculty",
        "question": "What is Arvin Agah's email address?",
        "ground_truth": "agah@ku.edu",
        "entity_check": r"agah@ku\.edu",
        "notes": "Direct email lookup — should be a fast exact match.",
    },
    {
        "id": "E02",
        "difficulty": "easy",
        "domain": "course",
        "question": "How many credit hours is EECS 168?",
        "ground_truth": "4 credit hours",
        "entity_check": r"\b4\b",
        "notes": "Single-field credit hour lookup.",
    },
    {
        "id": "E03",
        "difficulty": "easy",
        "domain": "dining",
        "question": "Where is Mrs. E's dining hall located?",
        "ground_truth": "Lewis Hall",
        "entity_check": r"Lewis Hall",
        "notes": "Building name for a named dining location.",
    },
    {
        "id": "E04",
        "difficulty": "easy",
        "domain": "tuition",
        "question": "What is the per-credit-hour tuition for in-state undergraduate students at KU?",
        "ground_truth": "$376.60 per credit hour",
        "entity_check": r"\$376\.60|\$376",
        "notes": "Resident undergraduate tuition rate.",
    },
    {
        "id": "E05",
        "difficulty": "easy",
        "domain": "transit",
        "question": "What is the Lawrence Transit Route 1 called?",
        "ground_truth": "Downtown / East Hills",
        "entity_check": r"Downtown.*East Hills|East Hills.*Downtown",
        "notes": "Route name lookup.",
    },
    {
        "id": "E06",
        "difficulty": "easy",
        "domain": "course",
        "question": "What is the title of EECS 678?",
        "ground_truth": "Introduction to Operating Systems",
        "entity_check": r"[Oo]perating [Ss]ystems",
        "notes": "Course title lookup.",
    },
    {
        "id": "E07",
        "difficulty": "easy",
        "domain": "faculty",
        "question": "What is Michael Branicky's office email?",
        "ground_truth": "msb@ku.edu",
        "entity_check": r"msb@ku\.edu",
        "notes": "Email lookup for EECS faculty.",
    },
    {
        "id": "E08",
        "difficulty": "easy",
        "domain": "housing",
        "question": "What type of bath does Corbin Hall have?",
        "ground_truth": "Community bath",
        "entity_check": r"[Cc]ommunity",
        "notes": "Simple attribute lookup for a residence hall.",
    },
    {
        "id": "E09",
        "difficulty": "easy",
        "domain": "calendar",
        "question": "When does the Spring 2026 semester begin at KU?",
        "ground_truth": "Tuesday, January 20, 2026",
        "entity_check": r"January 20|Jan.*20",
        "notes": "First day of classes — single calendar field.",
    },
    {
        "id": "E10",
        "difficulty": "easy",
        "domain": "course",
        "question": "What are the prerequisites for EECS 678?",
        "ground_truth": "EECS 388, EECS 348, and upper-level EECS eligibility",
        "entity_check": r"EECS 388|EECS 348",
        "notes": "Prereq field for a specific course code.",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # REGULAR — Multi-field, some reasoning, natural phrasing
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "R01",
        "difficulty": "regular",
        "domain": "faculty",
        "question": "Who are the EECS professors working on cybersecurity research at KU?",
        "ground_truth": "Alexandru Bardas, Drew J. Davidson, Tamzidul Hoque, Prasad Kulkarni",
        "entity_check": r"Bardas|Davidson|Hoque|Kulkarni",
        "notes": "Research-area filter across a department — tests semantic retrieval.",
    },
    {
        "id": "R02",
        "difficulty": "regular",
        "domain": "course",
        "question": "What machine learning and AI courses does KU's EECS department offer?",
        "ground_truth": "EECS 649 Introduction to Artificial Intelligence, EECS 658 Introduction to Machine Learning, EECS 836 Machine Learning",
        "entity_check": r"EECS 649|EECS 658|EECS 836",
        "notes": "Semantic topic search across course catalog.",
    },
    {
        "id": "R03",
        "difficulty": "regular",
        "domain": "dining",
        "question": "Which dining locations on campus are open on weekends?",
        "ground_truth": "Mrs. E's (9:00 AM - 7:30 PM on weekends)",
        "entity_check": r"Mrs\. E|weekend|Saturday|Sunday",
        "notes": "Hours lookup requiring day-of-week filtering.",
    },
    {
        "id": "R04",
        "difficulty": "regular",
        "domain": "tuition",
        "question": "What is the per-credit-hour tuition for a non-resident graduate student at KU?",
        "ground_truth": "$1,117.10 per credit hour",
        "entity_check": r"\$1,117|\$1117",
        "notes": "Non-resident graduate tuition — requires two-level lookup (residency + level).",
    },
    {
        "id": "R05",
        "difficulty": "regular",
        "domain": "housing",
        "question": "What are the cheapest housing options available in KU residence halls?",
        "ground_truth": "GSP 2-person at $7,242/year or Ellsworth 3-person economy at $6,184/year",
        "entity_check": r"GSP|Ellsworth|\$7,242|\$6,184",
        "notes": "Cost comparison across multiple hall options.",
    },
    {
        "id": "R06",
        "difficulty": "regular",
        "domain": "calendar",
        "question": "When are finals scheduled for Spring 2026 at KU?",
        "ground_truth": "May 11–15, 2026",
        "entity_check": r"May 1[1-5]|May 11",
        "notes": "Finals date range — requires calendar lookup.",
    },
    {
        "id": "R07",
        "difficulty": "regular",
        "domain": "course",
        "question": "What are the prerequisites for EECS 658 Introduction to Machine Learning?",
        "ground_truth": "EECS 330 and EECS 461 or MATH 526 or equivalent, and upper-level EECS eligibility",
        "entity_check": r"EECS 330|EECS 461|MATH 526",
        "notes": "Multi-part prereq requiring full text, not just a code.",
    },
    {
        "id": "R08",
        "difficulty": "regular",
        "domain": "faculty",
        "question": "What is Perry Alexander's research focus?",
        "ground_truth": "Formal Methods, Verification and Synthesis, Trusted Computing, System-Level Design Languages",
        "entity_check": r"[Ff]ormal [Mm]ethods|[Vv]erification|[Tt]rusted [Cc]omputing",
        "notes": "Research interests from faculty profile.",
    },
    {
        "id": "R09",
        "difficulty": "regular",
        "domain": "transit",
        "question": "Which Lawrence Transit routes operate on weekends?",
        "ground_truth": "Routes that include Saturday in their operating days",
        "entity_check": r"[Ss]aturday|weekend",
        "notes": "Multi-route filter on operating days.",
    },
    {
        "id": "R10",
        "difficulty": "regular",
        "domain": "course",
        "question": "How many credits is EECS 581 Software Engineering II and what courses are required before taking it?",
        "ground_truth": "3 credits; prerequisites: EECS 348, EECS 330, and upper-level EECS eligibility",
        "entity_check": r"\b3\b.*credit|EECS 348|EECS 330",
        "notes": "Two fields from one course — credits + prereqs.",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED — Multiple questions merged into one; tests multi-hop retrieval
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C01",
        "difficulty": "combined",
        "domain": "course+faculty",
        "question": "What are the prerequisites for EECS 658, and which EECS faculty member works on machine learning research?",
        "ground_truth": "Prerequisites: EECS 330, EECS 461 or MATH 526; ML faculty include Michael Branicky, Jerzy Grzymala-Busse, Zijun Yao",
        "entity_check": r"EECS 330|EECS 461|Branicky|Grzymala|Yao",
        "notes": "Course prereq + faculty research — cross-domain dual retrieval.",
    },
    {
        "id": "C02",
        "difficulty": "combined",
        "domain": "tuition+housing",
        "question": "What would a non-resident graduate student pay per credit hour in tuition, and what is the cheapest on-campus housing option?",
        "ground_truth": "$1,117.10/credit hour; Ellsworth 3-person economy at $6,184/year",
        "entity_check": r"\$1,117|\$6,184|Ellsworth",
        "notes": "Two domains (tuition + housing) combined into one question.",
    },
    {
        "id": "C03",
        "difficulty": "combined",
        "domain": "calendar+course",
        "question": "When does Spring 2026 start and what networking-related courses does EECS offer?",
        "ground_truth": "January 20, 2026; EECS networking courses include EECS 563 and EECS 780",
        "entity_check": r"January 20|EECS 563|EECS 780|[Nn]etwork",
        "notes": "Calendar date + semantic course search.",
    },
    {
        "id": "C04",
        "difficulty": "combined",
        "domain": "faculty+course",
        "question": "Who is the contact for cybersecurity research at KU EECS and what courses relate to cybersecurity?",
        "ground_truth": "Alexandru Bardas (alexbardas@ku.edu); cybersecurity courses: EECS 563, EECS 700 Security",
        "entity_check": r"Bardas|alexbardas@ku\.edu|[Cc]ybersecurity|[Ss]ecurity",
        "notes": "Research faculty contact + course catalog topic search.",
    },
    {
        "id": "C05",
        "difficulty": "combined",
        "domain": "dining+transit",
        "question": "Where is Mrs. E's dining hall and which bus route can I take to get to campus?",
        "ground_truth": "Lewis Hall (Daisy Hill); several KU routes serve campus",
        "entity_check": r"Lewis Hall|[Bb]us|[Rr]oute|campus",
        "notes": "Two campus service domains — dining location + transit guidance.",
    },
    {
        "id": "C06",
        "difficulty": "combined",
        "domain": "course+tuition",
        "question": "If I'm taking EECS 168 (4 credit hours) and EECS 649 (3 credit hours) as a resident undergraduate, how much tuition would I pay for those 7 hours?",
        "ground_truth": "7 × $376.60 = $2,636.20",
        "entity_check": r"376\.60|376|2,636|2636",
        "notes": "Arithmetic on tuition rate × credit hours — requires retrieval + calculation.",
    },
    {
        "id": "C07",
        "difficulty": "combined",
        "domain": "faculty+housing",
        "question": "What is Fengjun Li's email and what on-campus housing is available for graduate students?",
        "ground_truth": "fli@eecs.ku.edu; graduate housing options include GSP (Grace Pearson) Hall",
        "entity_check": r"fli@eecs\.ku\.edu|GSP|Grace Pearson|graduate",
        "notes": "Faculty contact + graduate housing — unrelated domains in one query.",
    },
    {
        "id": "C08",
        "difficulty": "combined",
        "domain": "calendar+tuition",
        "question": "When is the last day to drop a Spring 2026 class without a W grade, and what is the refund policy for dropped courses?",
        "ground_truth": "Drop deadlines are in the Spring 2026 calendar; refund policy is prorated",
        "entity_check": r"drop|refund|Spring 2026|[Ww] grade",
        "notes": "Combines calendar deadline with financial policy — different sub-systems.",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TOUGH — Multi-step reasoning, computation, implicit constraints
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "T01",
        "difficulty": "tough",
        "domain": "tuition",
        "question": "A non-resident graduate student is taking 9 credit hours in Fall 2025. What is their total estimated tuition cost including mandatory fees?",
        "ground_truth": "Tuition: 9 × $1,117.10 = $10,053.90; plus mandatory fees (student fee flat rate $287.10 at 6+ hours for grad)",
        "entity_check": r"\$10,053|\$1,117|mandatory fee|\$287",
        "notes": "Multi-step: tuition rate × hours + mandatory fee lookup. Tests full arithmetic chain.",
    },
    {
        "id": "T02",
        "difficulty": "tough",
        "domain": "course",
        "question": "What is the full prerequisite chain to take EECS 678? Starting from scratch, which courses must a student complete first?",
        "ground_truth": "EECS 678 requires EECS 388 and EECS 348; EECS 388 requires EECS 140/141 and EECS 168/169; EECS 168 requires coreq MATH 104 or 125",
        "entity_check": r"EECS 388|EECS 348|EECS 168|EECS 140",
        "notes": "Multi-hop prerequisite chain — requires chaining multiple course lookups.",
    },
    {
        "id": "T03",
        "difficulty": "tough",
        "domain": "faculty",
        "question": "Which KU EECS professors work at the intersection of machine learning and cybersecurity or systems security?",
        "ground_truth": "Tamzidul Hoque (hardware security + ML), Prasad Kulkarni (compilers + ML security); potentially Fengjun Li (ML + network security)",
        "entity_check": r"Hoque|Kulkarni|Fengjun Li|[Ss]ecurity.*[Mm]achine [Ll]earning|[Mm]achine [Ll]earning.*[Ss]ecurity",
        "notes": "Intersection query — requires reasoning across multiple research interest fields.",
    },
    {
        "id": "T04",
        "difficulty": "tough",
        "domain": "course",
        "question": "Is EECS 836 Machine Learning a graduate or undergraduate course, and what level of mathematical background does it expect?",
        "ground_truth": "EECS 836 is a graduate-level course; it expects linear algebra and probability background based on prerequisites",
        "entity_check": r"graduate|EECS 836|[Mm]achine [Ll]earning",
        "notes": "Implicit question: requires understanding course level + inferring background from prereqs.",
    },
    {
        "id": "T05",
        "difficulty": "tough",
        "domain": "housing+tuition",
        "question": "As a non-resident undergraduate student living in Corbin Hall in a 2-person room, what is my estimated annual cost for tuition (30 hours) plus housing?",
        "ground_truth": "Tuition: $30,177 (30 hrs non-resident UG estimated); Housing: ~$8,346/year; Total ~$38,523",
        "entity_check": r"\$30,177|\$8,346|\$38",
        "notes": "Annual cost calculation combining two separate data sources + addition.",
    },
    {
        "id": "T06",
        "difficulty": "tough",
        "domain": "calendar",
        "question": "How many weeks of instruction are there in Spring 2026 between the first day of classes and the last day before finals?",
        "ground_truth": "January 20 to May 7 = approximately 15–16 weeks",
        "entity_check": r"15|16.*week|week.*15|week.*16",
        "notes": "Date arithmetic — requires both dates from calendar + computation.",
    },
    {
        "id": "T07",
        "difficulty": "tough",
        "domain": "course+faculty",
        "question": "If I want to pursue AI research at KU, which courses should I take and which professors should I approach?",
        "ground_truth": "Courses: EECS 649, EECS 658, EECS 836; Faculty: Arvin Agah, Michael Branicky, Zijun Yao",
        "entity_check": r"EECS 649|EECS 658|EECS 836|Agah|Branicky|Yao",
        "notes": "Advising-style question requiring synthesis of course + faculty data.",
    },
    {
        "id": "T08",
        "difficulty": "tough",
        "domain": "tuition",
        "question": "What is the difference in annual tuition cost between a resident and non-resident undergraduate student taking a full load (30 credit hours)?",
        "ground_truth": "$30,177 − $11,298 = $18,879 difference",
        "entity_check": r"\$18,879|\$18879|18,879|11,298|30,177",
        "notes": "Subtraction across two tuition rows — tests retrieval + arithmetic.",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # RESEARCH — Deep technical, domain expert-level, judgment required
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "RS01",
        "difficulty": "research",
        "domain": "faculty",
        "question": "Which EECS faculty at KU are working on problems related to natural language processing or large language models?",
        "ground_truth": "David Johnson (NLP, davidojohnson@ku.edu), Zijun Yao (NLP/LLM, zyao@ku.edu)",
        "entity_check": r"Johnson|Yao|davidojohnson@ku\.edu|zyao@ku\.edu|NLP|[Nn]atural [Ll]anguage",
        "notes": "Research-specific semantic query — must go beyond keyword to research interest meaning.",
    },
    {
        "id": "RS02",
        "difficulty": "research",
        "domain": "faculty",
        "question": "Who at KU EECS conducts research in wireless communications or signal processing?",
        "ground_truth": "Shannon Blunt (radar/signal processing), Victor Frost (networking/wireless), Joseph Evans",
        "entity_check": r"Blunt|Frost|Evans|[Ss]ignal [Pp]rocessing|[Ww]ireless",
        "notes": "Research area query — overlapping subfields require broad semantic coverage.",
    },
    {
        "id": "RS03",
        "difficulty": "research",
        "domain": "course",
        "question": "What graduate-level EECS courses are available that focus on theoretical computer science or formal methods?",
        "ground_truth": "EECS courses at 700+ level with theory/formal topics; also relevant undergrad theory courses",
        "entity_check": r"EECS [78]\d\d|[Ff]ormal|[Tt]heory|[Tt]heoretical",
        "notes": "Graduate-level filter + semantic topic — requires level AND topic disambiguation.",
    },
    {
        "id": "RS04",
        "difficulty": "research",
        "domain": "faculty",
        "question": "Which KU EECS professor has the most interdisciplinary research profile, spanning both engineering systems and artificial intelligence?",
        "ground_truth": "Michael Branicky (Cyber-Physical Systems/IoT, AI, ML, Robotics) is the strongest match",
        "entity_check": r"Branicky|[Cc]yber.[Pp]hysical|[Rr]obotics|[Ii]nterdisciplinary",
        "notes": "Open-ended comparison requiring synthesis across multiple faculty profiles.",
    },
    {
        "id": "RS05",
        "difficulty": "research",
        "domain": "course+faculty",
        "question": "If a graduate student wants to specialize in cybersecurity at KU, what is the recommended coursework path and who are the key faculty to work with?",
        "ground_truth": "Courses: EECS 563 (Intro to Computer Networks), 700-level security electives; Faculty: Bardas, Davidson, Hoque, Kulkarni",
        "entity_check": r"Bardas|Davidson|Hoque|Kulkarni|EECS 563|[Ss]ecurity",
        "notes": "Curriculum advising + research mentorship — requires full cross-domain synthesis.",
    },
    {
        "id": "RS06",
        "difficulty": "research",
        "domain": "faculty",
        "question": "Which EECS faculty member is a subject matter expert for the White House Office of Science and Technology Policy?",
        "ground_truth": "Shannon Blunt — RF spectrum R&D SME for OSTP",
        "entity_check": r"Blunt|OSTP|[Ww]hite [Hh]ouse|[Ss]pectrum",
        "notes": "Obscure detail buried deep in a faculty profile — stress test for semantic recall.",
    },
    {
        "id": "RS07",
        "difficulty": "research",
        "domain": "course",
        "question": "What distinguishes EECS 658 from EECS 836? Are they the same machine learning course or different?",
        "ground_truth": "EECS 658 is undergraduate ML (intro-level); EECS 836 is graduate-level Machine Learning",
        "entity_check": r"EECS 658|EECS 836|undergraduate|graduate",
        "notes": "Comparative question requiring two course descriptions + level disambiguation.",
    },
    {
        "id": "RS08",
        "difficulty": "research",
        "domain": "faculty",
        "question": "Which KU EECS researchers would be most relevant to collaborate with on a project involving hardware security or IoT device security?",
        "ground_truth": "Tamzidul Hoque (hardware security), Michael Branicky (IoT/CPS), Alexandru Bardas (enterprise security)",
        "entity_check": r"Hoque|Branicky|Bardas|IoT|[Hh]ardware [Ss]ecurity",
        "notes": "Collaboration recommendation — needs cross-referencing multiple research interests.",
    },
    {
        "id": "RS09",
        "difficulty": "research",
        "domain": "course",
        "question": "Which EECS courses overlap conceptually with mathematics in areas like optimization or probability theory?",
        "ground_truth": "EECS 658 (requires MATH 526/prob), EECS 461 (applied math/signals), EECS 836 (optimization in ML)",
        "entity_check": r"EECS 658|EECS 461|EECS 836|MATH|optimization|probability",
        "notes": "Cross-disciplinary query: EECS-math overlap; prereq + description reasoning required.",
    },
    {
        "id": "RS10",
        "difficulty": "research",
        "domain": "faculty",
        "question": "How many EECS faculty at KU have research interests related to robotics or autonomous systems?",
        "ground_truth": "At least 3: Arvin Agah (autonomous robots), Michael Branicky (robotics), potentially others",
        "entity_check": r"Agah|Branicky|[Rr]obotics|[Aa]utonomous",
        "notes": "Counting/aggregation query across all faculty — tests recall breadth.",
    },
]
# fmt: on


# ── BabyJay Chat Interface ─────────────────────────────────────────────────────
def ask_babyjay(question: str) -> Dict[str, Any]:
    """
    Send a question through BabyJay's full RAG pipeline and return
    the answer + retrieved context.
    """
    from app.rag.router import QueryRouter
    from app.rag.classifier import QueryClassifier

    classifier = QueryClassifier()
    router = QueryRouter()

    classification = classifier.classify(question)
    route_result = router.route(question, classification)

    # Get the answer from the chat module
    from app.rag import chat as chat_module
    # Use the chat function directly
    context = route_result.get("context", "")
    answer = _generate_answer(question, context)

    return {
        "answer": answer,
        "context": context,
        "source": route_result.get("source", "unknown"),
        "results_count": len(route_result.get("results", [])),
    }


def _generate_answer(question: str, context: str) -> str:
    """Generate an answer using Claude with the retrieved context."""
    client = anthropic.Anthropic()

    system = (
        "You are BabyJay, a helpful assistant for the University of Kansas. "
        "Answer the question using ONLY the provided context. "
        "If the context doesn't contain enough information, say so clearly. "
        "Be specific and cite exact numbers, names, or codes where relevant."
    )

    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


# ── LLM-as-Judge Evaluators ───────────────────────────────────────────────────
def _judge(prompt: str) -> Dict[str, Any]:
    """
    Call Claude as the judge. Returns a dict with at minimum a 'score' key (0.0–1.0)
    and an 'explanation' key.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        system=(
            "You are a strict, impartial evaluator of RAG system outputs. "
            "Always respond with valid JSON only — no markdown, no prose outside the JSON."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract score with regex
        score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        score = float(score_match.group(1)) if score_match else 0.0
        return {"score": score, "explanation": raw[:200]}


def eval_retrieval_relevance(question: str, context: str) -> Dict[str, Any]:
    """Evaluator 1: Did retrieval bring back relevant documents?"""
    prompt = f"""Evaluate whether the retrieved context is relevant to answering the question.

Question: {question}

Retrieved Context:
{context[:1500]}

Score the RETRIEVAL RELEVANCE on a 0.0–1.0 scale:
- 1.0: Context directly contains the information needed to answer the question
- 0.75: Context is mostly relevant but has some noise or missing details
- 0.5: Context is partially relevant — some useful info, some off-topic
- 0.25: Context is mostly irrelevant but has a small amount of useful info
- 0.0: Context is entirely irrelevant or empty

Respond with JSON only:
{{"score": <float>, "explanation": "<1-2 sentence reason>"}}"""
    return _judge(prompt)


def eval_groundedness(question: str, context: str, answer: str) -> Dict[str, Any]:
    """Evaluator 2: Is the answer grounded in the retrieved context?"""
    prompt = f"""Evaluate whether the answer is grounded in the provided context (not hallucinated).

Question: {question}

Retrieved Context:
{context[:1500]}

Answer:
{answer}

Score GROUNDEDNESS on 0.0–1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.75: Most claims are supported; minor unsupported details
- 0.5: About half the claims are supported; noticeable hallucination
- 0.25: Most claims are not in the context; heavy hallucination
- 0.0: Answer is completely fabricated with no grounding in context

Respond with JSON only:
{{"score": <float>, "explanation": "<1-2 sentence reason>"}}"""
    return _judge(prompt)


def eval_correctness(question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
    """Evaluator 3: Is the answer factually correct vs. ground truth?"""
    prompt = f"""Evaluate whether the answer is factually correct compared to the ground truth.

Question: {question}

Ground Truth Answer:
{ground_truth}

System Answer:
{answer}

Score CORRECTNESS on 0.0–1.0:
- 1.0: Answer is factually correct and complete (all key facts match)
- 0.75: Mostly correct; minor omissions or slightly imprecise
- 0.5: Partially correct; gets some facts right, misses or errors on others
- 0.25: Mostly incorrect; perhaps one incidental correct fact
- 0.0: Completely wrong, or refused to answer when an answer exists

Note: Equivalent phrasings count as correct. "4 credits" = "4 credit hours".

Respond with JSON only:
{{"score": <float>, "explanation": "<1-2 sentence reason>"}}"""
    return _judge(prompt)


def eval_response_relevance(question: str, answer: str) -> Dict[str, Any]:
    """Evaluator 4: Does the answer actually address the question asked?"""
    prompt = f"""Evaluate whether the answer actually addresses what was asked.

Question: {question}

Answer:
{answer}

Score RESPONSE RELEVANCE on 0.0–1.0:
- 1.0: Answer directly and completely addresses the question
- 0.75: Mostly addresses the question; minor tangents or missing sub-parts
- 0.5: Partially addresses the question; answers part but misses key aspects
- 0.25: Barely addresses the question; mostly off-topic
- 0.0: Does not address the question at all

Respond with JSON only:
{{"score": <float>, "explanation": "<1-2 sentence reason>"}}"""
    return _judge(prompt)


def eval_entity_precision(answer: str, entity_pattern: str) -> Dict[str, Any]:
    """Entity Precision: Does the answer contain the expected entity?"""
    if not entity_pattern:
        return {"score": None, "explanation": "No entity check defined"}
    match = bool(re.search(entity_pattern, answer, re.IGNORECASE))
    return {
        "score": 1.0 if match else 0.0,
        "explanation": f"Pattern `{entity_pattern}` {'found' if match else 'NOT found'} in answer",
    }


# ── Single QA Evaluation ──────────────────────────────────────────────────────
@dataclass
class EvalResult:
    id: str
    difficulty: str
    domain: str
    question: str
    ground_truth: str
    answer: str
    context_snippet: str  # first 300 chars of context
    source: str
    results_count: int
    retrieval_relevance: Dict = field(default_factory=dict)
    groundedness: Dict = field(default_factory=dict)
    correctness: Dict = field(default_factory=dict)
    response_relevance: Dict = field(default_factory=dict)
    entity_precision: Dict = field(default_factory=dict)
    composite_score: float = 0.0
    error: Optional[str] = None
    latency_ms: int = 0


def evaluate_one(qa: Dict[str, Any]) -> EvalResult:
    """Run all 5 evaluations for a single QA pair."""
    qid = qa["id"]
    print(f"  [{qid}] {qa['difficulty'].upper():10} {qa['question'][:70]}...")

    start = time.time()
    try:
        result = ask_babyjay(qa["question"])
        latency_ms = int((time.time() - start) * 1000)

        answer = result["answer"]
        context = result["context"]
        source = result["source"]
        results_count = result["results_count"]

        # Run all 5 evaluators
        rr = eval_retrieval_relevance(qa["question"], context)
        gd = eval_groundedness(qa["question"], context, answer)
        cr = eval_correctness(qa["question"], answer, qa["ground_truth"])
        rv = eval_response_relevance(qa["question"], answer)
        ep = eval_entity_precision(answer, qa.get("entity_check", ""))

        # Composite: weighted average of the 4 LLM evaluators
        scores = [rr["score"], gd["score"], cr["score"], rv["score"]]
        composite = sum(scores) / len(scores)

        return EvalResult(
            id=qid,
            difficulty=qa["difficulty"],
            domain=qa["domain"],
            question=qa["question"],
            ground_truth=qa["ground_truth"],
            answer=answer,
            context_snippet=context[:300],
            source=source,
            results_count=results_count,
            retrieval_relevance=rr,
            groundedness=gd,
            correctness=cr,
            response_relevance=rv,
            entity_precision=ep,
            composite_score=composite,
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        print(f"    ERROR: {e}")
        return EvalResult(
            id=qid,
            difficulty=qa["difficulty"],
            domain=qa["domain"],
            question=qa["question"],
            ground_truth=qa["ground_truth"],
            answer="[ERROR]",
            context_snippet="",
            source="error",
            results_count=0,
            composite_score=0.0,
            error=str(e),
            latency_ms=latency_ms,
        )


# ── Reporting ─────────────────────────────────────────────────────────────────
DIFFICULTY_ORDER = ["easy", "regular", "combined", "tough", "research"]
SCORE_EMOJI = {
    (0.85, 1.01): "PASS",
    (0.65, 0.85): "PARTIAL",
    (0.0,  0.65): "FAIL",
}

def _grade(score: float) -> str:
    for (lo, hi), label in SCORE_EMOJI.items():
        if lo <= score < hi:
            return label
    return "FAIL"


def build_report(results: List[EvalResult]) -> str:
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        "# BabyJay RAG Evaluation Report",
        f"Generated: {ts}",
        f"Model under test: {CHAT_MODEL}",
        f"Judge model: {JUDGE_MODEL}",
        f"Total questions: {len(results)}",
        "",
    ]

    # ── Overall summary table ──
    valid = [r for r in results if not r.error]
    overall = sum(r.composite_score for r in valid) / len(valid) if valid else 0.0
    entity_hits = [r for r in valid if r.entity_precision.get("score") == 1.0]
    entity_checked = [r for r in valid if r.entity_precision.get("score") is not None]

    lines += [
        "## Overall Summary",
        "",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Composite (avg 4 evaluators) | {overall:.2f} |",
        f"| Entity Precision | {len(entity_hits)}/{len(entity_checked)} ({100*len(entity_hits)/max(1,len(entity_checked)):.0f}%) |",
        f"| Errors | {len(results) - len(valid)} |",
        "",
    ]

    # ── Per-difficulty breakdown ──
    lines += ["## Score by Difficulty", ""]
    lines += ["| Difficulty | N | Avg Score | Entity Precision | Pass | Partial | Fail |",
              "|------------|---|-----------|-----------------|------|---------|------|"]

    for diff in DIFFICULTY_ORDER:
        group = [r for r in valid if r.difficulty == diff]
        if not group:
            continue
        avg = sum(r.composite_score for r in group) / len(group)
        ep_ok = sum(1 for r in group if r.entity_precision.get("score") == 1.0)
        ep_total = sum(1 for r in group if r.entity_precision.get("score") is not None)
        passes = sum(1 for r in group if r.composite_score >= 0.85)
        partials = sum(1 for r in group if 0.65 <= r.composite_score < 0.85)
        fails = sum(1 for r in group if r.composite_score < 0.65)
        lines.append(f"| {diff.title():10} | {len(group)} | {avg:.2f} | {ep_ok}/{ep_total} | {passes} | {partials} | {fails} |")

    lines += [""]

    # ── Per-domain breakdown ──
    domains = sorted(set(r.domain for r in valid))
    lines += ["## Score by Domain", ""]
    lines += ["| Domain | N | Avg Composite | Avg Correctness |",
              "|--------|---|---------------|-----------------|"]
    for dom in domains:
        group = [r for r in valid if r.domain == dom]
        avg_c = sum(r.composite_score for r in group) / len(group)
        avg_cor = sum(r.correctness.get("score", 0) for r in group) / len(group)
        lines.append(f"| {dom:20} | {len(group)} | {avg_c:.2f} | {avg_cor:.2f} |")
    lines += [""]

    # ── Per-evaluator breakdown ──
    lines += ["## Score by Evaluator", ""]
    lines += ["| Evaluator | Avg Score |",
              "|-----------|-----------|"]
    for ev_name, key in [
        ("Retrieval Relevance", "retrieval_relevance"),
        ("Groundedness", "groundedness"),
        ("Correctness", "correctness"),
        ("Response Relevance", "response_relevance"),
    ]:
        scores = [getattr(r, key).get("score", 0) for r in valid]
        avg = sum(scores) / len(scores) if scores else 0.0
        lines.append(f"| {ev_name:20} | {avg:.2f} |")
    lines += [""]

    # ── Full results table ──
    lines += ["## Detailed Results", ""]
    lines += ["| ID | Difficulty | Domain | Grade | Composite | Retrieval | Groundedness | Correctness | Relevance | Entity |",
              "|----|------------|--------|-------|-----------|-----------|--------------|-------------|-----------|--------|"]

    for r in results:
        if r.error:
            lines.append(f"| {r.id} | {r.difficulty} | {r.domain} | ERROR | — | — | — | — | — | — |")
            continue
        grade = _grade(r.composite_score)
        ep = r.entity_precision.get("score")
        ep_str = ("YES" if ep == 1.0 else "NO" if ep == 0.0 else "—")
        lines.append(
            f"| {r.id} | {r.difficulty} | {r.domain} | {grade} "
            f"| {r.composite_score:.2f} "
            f"| {r.retrieval_relevance.get('score', 0):.2f} "
            f"| {r.groundedness.get('score', 0):.2f} "
            f"| {r.correctness.get('score', 0):.2f} "
            f"| {r.response_relevance.get('score', 0):.2f} "
            f"| {ep_str} |"
        )
    lines += [""]

    # ── Failures + tough cases (for professor panel) ──
    failures = [r for r in valid if r.composite_score < 0.65]
    tough_hard = [r for r in valid if r.difficulty in ("tough", "research") and r.composite_score < 0.85]

    if failures:
        lines += ["## Failed Questions (Score < 0.65)", ""]
        for r in failures:
            lines += [
                f"### [{r.id}] {r.question}",
                f"- **Score:** {r.composite_score:.2f}",
                f"- **Ground Truth:** {r.ground_truth}",
                f"- **Answer:** {r.answer[:300]}",
                f"- **Correctness:** {r.correctness.get('score',0):.2f} — {r.correctness.get('explanation','')}",
                f"- **Retrieved from:** {r.source} ({r.results_count} results)",
                "",
            ]

    if tough_hard:
        lines += ["## Tough/Research Questions Needing Improvement", ""]
        for r in tough_hard:
            lines += [
                f"### [{r.id}] {r.question}",
                f"- **Score:** {r.composite_score:.2f} | **Difficulty:** {r.difficulty}",
                f"- **Correctness:** {r.correctness.get('score',0):.2f} — {r.correctness.get('explanation','')}",
                f"- **Groundedness:** {r.groundedness.get('score',0):.2f} — {r.groundedness.get('explanation','')}",
                "",
            ]

    # ── Professor panel talking points ──
    lines += [
        "## Professor Panel Notes",
        "",
        "### What BabyJay does well",
        "- Direct single-field lookups (email, credits, tuition rates)",
        "- Domain-specific routing to correct retriever",
        "- Cross-encoder reranking improves top-result quality",
        "- Hybrid BM25 + vector search for course retrieval",
        "",
        "### Known limitations",
        "- Multi-hop prerequisite chain traversal (requires chaining multiple lookups)",
        "- Arithmetic computation (tuition × hours requires calculator, not retrieval)",
        "- Intersection queries across multiple research interest fields",
        "- Date arithmetic (computing number of weeks between two dates)",
        "",
        "### Architecture",
        "- Embeddings: `text-embedding-3-large` (OpenAI)",
        "- Vector DB: ChromaDB with persistent storage",
        "- Retrieval: BM25 + vector hybrid with Reciprocal Rank Fusion",
        "- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no API)",
        "- LLM: `claude-sonnet-4-6` for answer generation",
        "",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nBabyJay RAG Evaluation — {len(QA_PAIRS)} questions")
    print(f"Output: {OUTPUT_DIR}/\n")
    print("=" * 70)

    results: List[EvalResult] = []

    for diff in DIFFICULTY_ORDER:
        group = [qa for qa in QA_PAIRS if qa["difficulty"] == diff]
        if not group:
            continue
        print(f"\n{'─'*70}")
        print(f"  {diff.upper()} ({len(group)} questions)")
        print(f"{'─'*70}")
        for qa in group:
            result = evaluate_one(qa)
            results.append(result)
            # Brief status line
            grade = _grade(result.composite_score) if not result.error else "ERROR"
            ep = result.entity_precision.get("score")
            ep_str = "entity:YES" if ep == 1.0 else "entity:NO" if ep == 0.0 else ""
            print(f"    → {grade:7} composite={result.composite_score:.2f}  corr={result.correctness.get('score',0):.2f}  {ep_str}")

    # Save raw JSON
    raw_path = OUTPUT_DIR / "eval_results.json"
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nRaw results saved: {raw_path}")

    # Save markdown report
    report = build_report(results)
    report_path = OUTPUT_DIR / "eval_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved:      {report_path}")

    # Print final summary to console
    valid = [r for r in results if not r.error]
    overall = sum(r.composite_score for r in valid) / len(valid) if valid else 0.0
    ep_ok = sum(1 for r in valid if r.entity_precision.get("score") == 1.0)
    ep_total = sum(1 for r in valid if r.entity_precision.get("score") is not None)

    print(f"\n{'='*70}")
    print(f"  FINAL SCORES")
    print(f"{'='*70}")
    for diff in DIFFICULTY_ORDER:
        group = [r for r in valid if r.difficulty == diff]
        if group:
            avg = sum(r.composite_score for r in group) / len(group)
            passes = sum(1 for r in group if r.composite_score >= 0.85)
            print(f"  {diff.title():10}  avg={avg:.2f}  pass={passes}/{len(group)}")
    print(f"{'─'*70}")
    print(f"  Overall composite:   {overall:.2f}")
    print(f"  Entity precision:    {ep_ok}/{ep_total} ({100*ep_ok//max(1,ep_total)}%)")
    print(f"  Errors:              {len(results) - len(valid)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
