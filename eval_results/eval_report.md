# BabyJay RAG Evaluation Report
Generated: 2026-04-10 01:14
Model under test: claude-sonnet-4-6
Judge model: claude-sonnet-4-6
Total questions: 46

## Overall Summary

| Metric | Score |
|--------|-------|
| Composite (avg 4 evaluators) | 0.72 |
| Entity Precision | 43/46 (93%) |
| Errors | 0 |

## Score by Difficulty

| Difficulty | N | Avg Score | Entity Precision | Pass | Partial | Fail |
|------------|---|-----------|-----------------|------|---------|------|
| Easy       | 10 | 0.98 | 10/10 | 10 | 0 | 0 |
| Regular    | 10 | 0.85 | 9/10 | 5 | 4 | 1 |
| Combined   | 8 | 0.54 | 7/8 | 0 | 1 | 7 |
| Tough      | 8 | 0.59 | 7/8 | 1 | 3 | 4 |
| Research   | 10 | 0.59 | 10/10 | 0 | 6 | 4 |

## Score by Domain

| Domain | N | Avg Composite | Avg Correctness |
|--------|---|---------------|-----------------|
| calendar             | 3 | 0.77 | 0.67 |
| calendar+course      | 1 | 0.61 | 0.50 |
| calendar+tuition     | 1 | 0.50 | 0.50 |
| course               | 11 | 0.81 | 0.70 |
| course+faculty       | 3 | 0.47 | 0.63 |
| course+tuition       | 1 | 0.42 | 0.00 |
| dining               | 2 | 0.91 | 0.75 |
| dining+transit       | 1 | 0.62 | 0.75 |
| faculty              | 11 | 0.69 | 0.71 |
| faculty+course       | 1 | 0.51 | 0.50 |
| faculty+housing      | 1 | 0.33 | 0.10 |
| housing              | 2 | 0.86 | 0.62 |
| housing+tuition      | 1 | 0.65 | 0.50 |
| transit              | 2 | 0.84 | 0.75 |
| tuition              | 4 | 0.90 | 0.81 |
| tuition+housing      | 1 | 0.55 | 0.50 |

## Score by Evaluator

| Evaluator | Avg Score |
|-----------|-----------|
| Retrieval Relevance  | 0.66 |
| Groundedness         | 0.76 |
| Correctness          | 0.66 |
| Response Relevance   | 0.82 |

## Detailed Results

| ID | Difficulty | Domain | Grade | Composite | Retrieval | Groundedness | Correctness | Relevance | Entity |
|----|------------|--------|-------|-----------|-----------|--------------|-------------|-----------|--------|
| E01 | easy | faculty | PASS | 0.91 | 1.00 | 1.00 | 0.75 | 0.90 | YES |
| E02 | easy | course | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| E03 | easy | dining | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| E04 | easy | tuition | PASS | 0.90 | 1.00 | 1.00 | 0.75 | 0.85 | YES |
| E05 | easy | transit | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| E06 | easy | course | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| E07 | easy | faculty | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| E08 | easy | housing | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| E09 | easy | calendar | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| E10 | easy | course | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| R01 | regular | faculty | FAIL | 0.62 | 0.75 | 0.25 | 0.50 | 1.00 | YES |
| R02 | regular | course | PARTIAL | 0.78 | 0.75 | 1.00 | 0.50 | 0.85 | YES |
| R03 | regular | dining | PARTIAL | 0.82 | 0.90 | 0.90 | 0.50 | 1.00 | YES |
| R04 | regular | tuition | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| R05 | regular | housing | PARTIAL | 0.72 | 1.00 | 0.70 | 0.25 | 0.95 | NO |
| R06 | regular | calendar | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| R07 | regular | course | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| R08 | regular | faculty | PASS | 0.95 | 1.00 | 1.00 | 0.90 | 0.90 | YES |
| R09 | regular | transit | PARTIAL | 0.69 | 0.75 | 0.50 | 0.50 | 1.00 | YES |
| R10 | regular | course | PASS | 0.95 | 1.00 | 1.00 | 0.90 | 0.90 | YES |
| C01 | combined | course+faculty | PARTIAL | 0.74 | 0.75 | 0.20 | 1.00 | 1.00 | YES |
| C02 | combined | tuition+housing | FAIL | 0.55 | 0.50 | 0.25 | 0.50 | 0.95 | YES |
| C03 | combined | calendar+course | FAIL | 0.61 | 0.25 | 0.95 | 0.50 | 0.75 | YES |
| C04 | combined | faculty+course | FAIL | 0.51 | 0.50 | 0.40 | 0.50 | 0.62 | YES |
| C05 | combined | dining+transit | FAIL | 0.62 | 0.25 | 1.00 | 0.75 | 0.50 | YES |
| C06 | combined | course+tuition | FAIL | 0.42 | 0.25 | 0.95 | 0.00 | 0.50 | NO |
| C07 | combined | faculty+housing | FAIL | 0.33 | 0.20 | 0.50 | 0.10 | 0.50 | YES |
| C08 | combined | calendar+tuition | FAIL | 0.50 | 0.00 | 1.00 | 0.50 | 0.50 | YES |
| T01 | tough | tuition | PARTIAL | 0.71 | 0.75 | 0.75 | 0.50 | 0.85 | YES |
| T02 | tough | course | FAIL | 0.56 | 0.25 | 1.00 | 0.50 | 0.50 | YES |
| T03 | tough | faculty | FAIL | 0.42 | 0.25 | 0.20 | 0.40 | 0.85 | YES |
| T04 | tough | course | PARTIAL | 0.75 | 0.75 | 1.00 | 0.50 | 0.75 | YES |
| T05 | tough | housing+tuition | PARTIAL | 0.65 | 0.50 | 0.95 | 0.50 | 0.65 | YES |
| T06 | tough | calendar | FAIL | 0.31 | 0.00 | 1.00 | 0.00 | 0.25 | NO |
| T07 | tough | course+faculty | FAIL | 0.31 | 0.25 | 0.10 | 0.40 | 0.50 | YES |
| T08 | tough | tuition | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES |
| RS01 | research | faculty | PARTIAL | 0.70 | 0.50 | 0.40 | 1.00 | 0.90 | YES |
| RS02 | research | faculty | PARTIAL | 0.75 | 0.75 | 0.35 | 1.00 | 0.90 | YES |
| RS03 | research | course | FAIL | 0.31 | 0.00 | 1.00 | 0.00 | 0.25 | YES |
| RS04 | research | faculty | FAIL | 0.47 | 0.25 | 0.05 | 0.75 | 0.85 | YES |
| RS05 | research | course+faculty | FAIL | 0.36 | 0.25 | 0.20 | 0.50 | 0.50 | YES |
| RS06 | research | faculty | FAIL | 0.40 | 0.10 | 1.00 | 0.00 | 0.50 | YES |
| RS07 | research | course | PARTIAL | 0.81 | 0.75 | 0.85 | 0.75 | 0.90 | YES |
| RS08 | research | faculty | PARTIAL | 0.69 | 0.75 | 0.25 | 0.75 | 1.00 | YES |
| RS09 | research | course | PARTIAL | 0.76 | 0.75 | 0.85 | 0.50 | 0.95 | YES |
| RS10 | research | faculty | PARTIAL | 0.69 | 0.75 | 0.25 | 0.75 | 1.00 | YES |

## Failed Questions (Score < 0.65)

### [R01] Who are the EECS professors working on cybersecurity research at KU?
- **Score:** 0.62
- **Ground Truth:** Alexandru Bardas, Drew J. Davidson, Tamzidul Hoque, Prasad Kulkarni
- **Answer:** Based on the provided context, here are the EECS professors at KU whose research interests include cybersecurity-related topics:

1. **Alexandru Bardas** (alexbardas@ku.edu, 2040 Eaton Hall)
   - Cybersecurity from a systems perspective, Moving target defenses, Enterprise network security, Bringing 
- **Correctness:** 0.50 — The system answer includes all four professors from the ground truth (Alexandru Bardas, Drew J. Davidson, Tamzidul Hoque, Prasad Kulkarni) but also adds four additional professors (Fengjun Li, Bo Luo, Hossein Saiedian, Sumaiya Shomaji) not in the ground truth, making the answer partially correct but significantly expanded beyond the expected answer.
- **Retrieved from:** faculty_searcher (30 results)

### [C02] What would a non-resident graduate student pay per credit hour in tuition, and what is the cheapest on-campus housing option?
- **Score:** 0.55
- **Ground Truth:** $1,117.10/credit hour; Ellsworth 3-person economy at $6,184/year
- **Answer:** ## Non-Resident Graduate Tuition & Cheapest Housing

### Graduate Tuition (Non-Resident)
A non-resident graduate student pays **$1,117.10 per credit hour** in tuition. Note that additional course fees may apply depending on your program (for example, Engineering students pay an extra $95.00/credit h
- **Correctness:** 0.50 — The non-resident graduate tuition of $1,117.10 per credit hour is correct, but the cheapest on-campus housing option is wrong. The ground truth specifies Ellsworth 3-person economy at $6,184/year, while the system answer claims scholarship halls (Miller/Watkins) at $3,568/year as the cheapest option.
- **Retrieved from:** vector_housing, vector_tuition (20 results)

### [C03] When does Spring 2026 start and what networking-related courses does EECS offer?
- **Score:** 0.61
- **Ground Truth:** January 20, 2026; EECS networking courses include EECS 563 and EECS 780
- **Answer:** ## Spring 2026 Start Date

I don't have information about when Spring 2026 begins in the provided context. For accurate registration and semester dates, I'd recommend checking the **KU Academic Calendar** at [ku.edu](https://ku.edu) or contacting the Registrar's office.

---

## EECS Networking-Rela
- **Correctness:** 0.50 — The system answer correctly identifies EECS 563 and EECS 780 as networking courses (matching the ground truth), but fails to provide the Spring 2026 start date of January 20, 2026, claiming it doesn't have that information. Additionally, the system answer includes extra courses beyond what was asked, which doesn't detract from correctness but the missing start date results in a partial score.
- **Retrieved from:** course_vector (10 results)

### [C04] Who is the contact for cybersecurity research at KU EECS and what courses relate to cybersecurity?
- **Score:** 0.51
- **Ground Truth:** Alexandru Bardas (alexbardas@ku.edu); cybersecurity courses: EECS 563, EECS 700 Security
- **Answer:** ## Cybersecurity Research at KU EECS

### Faculty Contacts
The Cybersecurity research cluster has **10 associated faculty members**. Here are their contacts:

| Faculty Member | Email | Office |
|---|---|---|
| Perry Alexander | palexand@ku.edu | 2022 Eaton Hall |
| Alexandru Bardas | alexbardas@ku.
- **Correctness:** 0.50 — The system answer correctly identifies Alexandru Bardas and his email (alexbardas@ku.edu), but lists 9 additional faculty members not requested and fails to provide the cybersecurity courses (EECS 563, EECS 700 Security) that are part of the ground truth answer, instead claiming the context doesn't include course listings.
- **Retrieved from:** eecs_resources_retriever (3 results)

### [C05] Where is Mrs. E's dining hall and which bus route can I take to get to campus?
- **Score:** 0.62
- **Ground Truth:** Lewis Hall (Daisy Hill); several KU routes serve campus
- **Answer:** Based on the provided context, **Mrs. E's** is located in **Lewis Hall** in the **Daisy Hill** area of campus. It is an all-you-care-to-eat dining hall.

However, I don't have any information about bus routes in the provided context, so I'm unable to answer that part of your question. I'd recommend 
- **Correctness:** 0.75 — The answer correctly identifies Mrs. E's location as Lewis Hall in the Daisy Hill area, matching the ground truth. However, it fails to mention that several KU routes serve campus, instead claiming no bus route information is available in the context.
- **Retrieved from:** vector_dining (5 results)

### [C06] If I'm taking EECS 168 (4 credit hours) and EECS 649 (3 credit hours) as a resident undergraduate, how much tuition would I pay for those 7 hours?
- **Score:** 0.42
- **Ground Truth:** 7 × $376.60 = $2,636.20
- **Answer:** I appreciate the specific question, but I'm unable to calculate your tuition cost based on the provided context. The course information I have only includes details about:

- **EECS 168**: 4 credit hours
- **EECS 649**: 3 credit hours (graduate-level course)

The context does **not include any tuiti
- **Correctness:** 0.00 — The system refused to provide a tuition calculation, claiming no tuition rate information was available, while the ground truth shows a clear answer of $2,636.20 based on a rate of $376.60 per credit hour. The system completely failed to answer the question.
- **Retrieved from:** course_retriever (2 results)

### [C07] What is Fengjun Li's email and what on-campus housing is available for graduate students?
- **Score:** 0.33
- **Ground Truth:** fli@eecs.ku.edu; graduate housing options include GSP (Grace Pearson) Hall
- **Answer:** I can only partially answer your question based on the available context.

**Fengjun Li's email:** I don't have any information about Fengjun Li in the provided context, so I cannot provide their email.

**On-campus housing for graduate students:** The context mentions that KU Housing & Residence Li
- **Correctness:** 0.10 — The system answer fails to provide Fengjun Li's email (fli@eecs.ku.edu) and does not mention GSP (Grace Pearson) Hall as the specific graduate housing option. Both key facts from the ground truth are missing or incorrect.
- **Retrieved from:** vector_housing (10 results)

### [C08] When is the last day to drop a Spring 2026 class without a W grade, and what is the refund policy for dropped courses?
- **Score:** 0.50
- **Ground Truth:** Drop deadlines are in the Spring 2026 calendar; refund policy is prorated
- **Answer:** I don't have information about the last day to drop a Spring 2026 class without a W grade or the refund policy for dropped courses in the context provided to me. The sources I have access to only contain course listing information (course names, credit hours, and catalog URLs).

For accurate and up-
- **Correctness:** 0.50 — The system answer correctly acknowledges it doesn't have the specific drop deadline date and refund policy details, and directs to appropriate resources. However, the ground truth indicates these answers exist (drop deadlines in Spring 2026 calendar; refund policy is prorated), meaning a correct answer was possible. The system partially aligns by admitting uncertainty rather than providing wrong facts, but fails to provide the actual information.
- **Retrieved from:** course_vector (10 results)

### [T02] What is the full prerequisite chain to take EECS 678? Starting from scratch, which courses must a student complete first?
- **Score:** 0.56
- **Ground Truth:** EECS 678 requires EECS 388 and EECS 348; EECS 388 requires EECS 140/141 and EECS 168/169; EECS 168 requires coreq MATH 104 or 125
- **Answer:** ## Prerequisites for EECS 678 - Introduction to Operating Systems

Based on the provided context, the **direct prerequisites** for EECS 678 are:

- **EECS 388**
- **EECS 348**
- **Upper-level EECS eligibility**

---

## ⚠️ Important Limitation

I can only tell you the **immediate prerequisites** lis
- **Correctness:** 0.50 — The system answer correctly identifies EECS 388 and EECS 348 as direct prerequisites for EECS 678, but explicitly states it cannot trace the full prerequisite chain and fails to provide the deeper prerequisites (EECS 140/141, EECS 168/169, and the MATH 104/125 corequisite for EECS 168) that the question asks for.
- **Retrieved from:** course_vector, course_retriever (11 results)

### [T03] Which KU EECS professors work at the intersection of machine learning and cybersecurity or systems security?
- **Score:** 0.42
- **Ground Truth:** Tamzidul Hoque (hardware security + ML), Prasad Kulkarni (compilers + ML security); potentially Fengjun Li (ML + network security)
- **Answer:** Based on the provided context, here are the KU EECS professors whose research intersects **machine learning and cybersecurity/systems security**:

---

### 1. **Fengjun Li** — `fli@eecs.ku.edu` | 2018 Eaton Hall
- **Adversarial Machine Learning**
- IoT Security and Privacy
- Network Security
- Secur
- **Correctness:** 0.40 — The answer correctly identifies Fengjun Li as a key professor at the intersection of ML and cybersecurity, which matches the ground truth. However, it misses Tamzidul Hoque (hardware security + ML) and Prasad Kulkarni (compilers + ML security), who are explicitly named in the ground truth, and instead adds Bo Luo and Michael Branicky who are not mentioned in the ground truth as fitting this intersection.
- **Retrieved from:** faculty_searcher (30 results)

### [T06] How many weeks of instruction are there in Spring 2026 between the first day of classes and the last day before finals?
- **Score:** 0.31
- **Ground Truth:** January 20 to May 7 = approximately 15–16 weeks
- **Answer:** I don't have information about the Spring 2026 academic calendar in the provided context. The context only contains course listings with credit hours and catalog URLs.

For accurate information about the Spring 2026 semester schedule, I'd recommend checking:
- **KU's official academic calendar** at 
- **Correctness:** 0.00 — The system refused to answer, claiming it lacked the necessary information, while the ground truth provides a specific answer of approximately 15–16 weeks (January 20 to May 7).
- **Retrieved from:** course_vector (10 results)

### [T07] If I want to pursue AI research at KU, which courses should I take and which professors should I approach?
- **Score:** 0.31
- **Ground Truth:** Courses: EECS 649, EECS 658, EECS 836; Faculty: Arvin Agah, Michael Branicky, Zijun Yao
- **Answer:** # AI Research at KU: Professors to Approach

Based on the provided context, I can identify **relevant faculty members** for AI research, but I should be transparent that **the context does not include specific course listings or recommendations**. Here are the professors you should consider approach
- **Correctness:** 0.40 — The answer correctly identifies Arvin Agah and Zijun Yao from the ground truth faculty list, but misses Michael Branicky and includes several extra professors not in the ground truth. The answer completely omits the required courses (EECS 649, EECS 658, EECS 836), which are a key part of the ground truth answer.
- **Retrieved from:** faculty_searcher (30 results)

### [RS03] What graduate-level EECS courses are available that focus on theoretical computer science or formal methods?
- **Score:** 0.31
- **Ground Truth:** EECS courses at 700+ level with theory/formal topics; also relevant undergrad theory courses
- **Answer:** Based on the provided context, I cannot answer that question. The context only covers **graduate funding information** for EECS students at KU, including details about assistantships (GTAs and GRAs) and fellowship opportunities.

It does **not** contain any information about specific graduate-level 
- **Correctness:** 0.00 — The system refused to answer the question, claiming the context only covers graduate funding information. The ground truth indicates there are EECS courses at 700+ level with theory/formal topics that should be referenced, so the system's complete refusal to provide any course information results in a score of 0.0.
- **Retrieved from:** eecs_resources_retriever (1 results)

### [RS04] Which KU EECS professor has the most interdisciplinary research profile, spanning both engineering systems and artificial intelligence?
- **Score:** 0.47
- **Ground Truth:** Michael Branicky (Cyber-Physical Systems/IoT, AI, ML, Robotics) is the strongest match
- **Answer:** Based on the provided context, **Professor Michael Branicky** (3046 Eaton Hall) stands out as having one of the most interdisciplinary research profiles, spanning:

- **Cyber-Physical Systems / IoT**
- **Artificial Intelligence**
- **Machine Learning**
- **Robotics**

His background is further reinf
- **Correctness:** 0.75 — The system answer correctly identifies Michael Branicky as the strongest match for the most interdisciplinary research profile spanning engineering systems and AI, which aligns with the ground truth. However, it muddies the conclusion by suggesting David Johnson might be 'arguably the most cross-disciplinary,' introducing ambiguity not present in the ground truth.
- **Retrieved from:** faculty_searcher (30 results)

### [RS05] If a graduate student wants to specialize in cybersecurity at KU, what is the recommended coursework path and who are the key faculty to work with?
- **Score:** 0.36
- **Ground Truth:** Courses: EECS 563 (Intro to Computer Networks), 700-level security electives; Faculty: Bardas, Davidson, Hoque, Kulkarni
- **Answer:** ## Cybersecurity Specialization at KU

Based on the provided context, I can identify the **key faculty** working in cybersecurity areas, but I must be transparent that **specific recommended coursework paths are not included in the provided information**.

---

### Key Faculty to Work With

#### Sch
- **Correctness:** 0.50 — The system answer correctly identifies the key faculty (Bardas, Davidson, Hoque, Kulkarni) mentioned in the ground truth, plus additional faculty not in the ground truth. However, it fails to provide the recommended coursework path (EECS 563 and 700-level security electives), claiming this information is not in the provided context, which results in a partial score.
- **Retrieved from:** faculty_searcher (30 results)

### [RS06] Which EECS faculty member is a subject matter expert for the White House Office of Science and Technology Policy?
- **Score:** 0.40
- **Ground Truth:** Shannon Blunt — RF spectrum R&D SME for OSTP
- **Answer:** The provided context does not contain information about EECS faculty members serving as subject matter experts for the White House Office of Science and Technology Policy. The context only includes general faculty information such as office locations, phone numbers, emails, and brief research intere
- **Correctness:** 0.00 — The system refused to provide an answer, claiming the information was not in the context, rather than identifying Shannon Blunt as the RF spectrum R&D SME for OSTP.
- **Retrieved from:** faculty_searcher (30 results)

## Tough/Research Questions Needing Improvement

### [T01] A non-resident graduate student is taking 9 credit hours in Fall 2025. What is their total estimated tuition cost including mandatory fees?
- **Score:** 0.71 | **Difficulty:** tough
- **Correctness:** 0.50 — The system answer correctly calculates the base tuition ($10,053.90) but uses different fee structures than the ground truth. The ground truth specifies a flat-rate student fee of $287.10 for 6+ credit hours for graduate students, while the system answer uses a per-credit-hour student fee ($23.24 × 9 = $209.16) plus a Wellness Fee and Infrastructure Fee, arriving at a different total fee amount ($481.56 vs the ground truth's implied $287.10). The tuition calculation is correct but the mandatory fees breakdown conflicts with the ground truth.
- **Groundedness:** 0.75 — The core calculations (tuition at $1,117.10/credit hour, infrastructure fee at $3.00/credit hour, wellness fee flat rate of $245.40, student fee at $23.24/credit hour for under 12 hours) are all directly supported by the context. However, the specific program fee examples cited in the notes (Engineering $95.00/credit hour, Business Masters $230.20/credit hour) are not found in the provided context and appear to be hallucinated details.

### [T02] What is the full prerequisite chain to take EECS 678? Starting from scratch, which courses must a student complete first?
- **Score:** 0.56 | **Difficulty:** tough
- **Correctness:** 0.50 — The system answer correctly identifies EECS 388 and EECS 348 as direct prerequisites for EECS 678, but explicitly states it cannot trace the full prerequisite chain and fails to provide the deeper prerequisites (EECS 140/141, EECS 168/169, and the MATH 104/125 corequisite for EECS 168) that the question asks for.
- **Groundedness:** 1.00 — All claims in the answer are directly supported by the retrieved context. The answer correctly identifies EECS 388, EECS 348, and upper-level EECS eligibility as the direct prerequisites for EECS 678, and accurately acknowledges that the context does not contain prerequisite information for EECS 388 or EECS 348, making no unsupported claims about the full chain.

### [T03] Which KU EECS professors work at the intersection of machine learning and cybersecurity or systems security?
- **Score:** 0.42 | **Difficulty:** tough
- **Correctness:** 0.40 — The answer correctly identifies Fengjun Li as a key professor at the intersection of ML and cybersecurity, which matches the ground truth. However, it misses Tamzidul Hoque (hardware security + ML) and Prasad Kulkarni (compilers + ML security), who are explicitly named in the ground truth, and instead adds Bo Luo and Michael Branicky who are not mentioned in the ground truth as fitting this intersection.
- **Groundedness:** 0.20 — Only Fengjun Li's claims are fully supported by the retrieved context. Bo Luo, Michael Branicky, Jerzy W Grzymala-Busse, and Zijun Yao are not present in the provided context at all, making most of the answer hallucinated.

### [T04] Is EECS 836 Machine Learning a graduate or undergraduate course, and what level of mathematical background does it expect?
- **Score:** 0.75 | **Difficulty:** tough
- **Correctness:** 0.50 — The system answer correctly identifies EECS 836 as a graduate-level course, but fails to mention the linear algebra and probability prerequisites that are part of the ground truth answer. The system claims these details are not in the context, which contradicts the ground truth.
- **Groundedness:** 1.00 — Every claim in the answer is directly supported by the context: the course is graduate-level in the School of Engineering, the prerequisite is graduate standing in CS or CoE or consent of instructor, and the answer correctly acknowledges that specific mathematical background details are not present in the provided context.

### [T05] As a non-resident undergraduate student living in Corbin Hall in a 2-person room, what is my estimated annual cost for tuition (30 hours) plus housing?
- **Score:** 0.65 | **Difficulty:** tough
- **Correctness:** 0.50 — The system answer correctly identifies the tuition of $30,177 for 30 hours at the non-resident rate, but fails to provide the correct housing cost for Corbin Hall (~$8,346/year) and therefore gives an incorrect total of $42,177 instead of ~$38,523. The system acknowledges uncertainty about Corbin Hall rates but uses an incorrect general estimate instead.
- **Groundedness:** 0.95 — Nearly all claims are directly supported by the context: the $1,005.90/credit hour non-resident rate, the $30,177 annual tuition for 30 hours, and the $12,000 room & board general estimate all appear in the retrieved sources. The answer also correctly acknowledges that Corbin Hall-specific rates are not in the context, which is accurate and transparent rather than hallucinated.

### [T06] How many weeks of instruction are there in Spring 2026 between the first day of classes and the last day before finals?
- **Score:** 0.31 | **Difficulty:** tough
- **Correctness:** 0.00 — The system refused to answer, claiming it lacked the necessary information, while the ground truth provides a specific answer of approximately 15–16 weeks (January 20 to May 7).
- **Groundedness:** 1.00 — The answer correctly acknowledges that the provided context only contains course listings and does not include Spring 2026 academic calendar information. No hallucinated facts are present; the answer appropriately defers to official sources.

### [T07] If I want to pursue AI research at KU, which courses should I take and which professors should I approach?
- **Score:** 0.31 | **Difficulty:** tough
- **Correctness:** 0.40 — The answer correctly identifies Arvin Agah and Zijun Yao from the ground truth faculty list, but misses Michael Branicky and includes several extra professors not in the ground truth. The answer completely omits the required courses (EECS 649, EECS 658, EECS 836), which are a key part of the ground truth answer.
- **Groundedness:** 0.10 — The answer fabricates numerous details not present in the context, including email addresses, research areas, and entire faculty entries (Arvin Agah's details, David Johnson, Jerzy Grzymala-Busse, Brian McClendon, Zijun Yao, Fengjun Li, Shawn Keshmiri). Only Karthik Srinivasan's generative AI and interpretable ML research areas are partially supported by the retrieved context; everything else is hallucinated.

### [RS01] Which EECS faculty at KU are working on problems related to natural language processing or large language models?
- **Score:** 0.70 | **Difficulty:** research
- **Correctness:** 1.00 — The system answer correctly identifies both faculty members (David Johnson and Zijun Yao) with their correct email addresses, matching the ground truth exactly. The additional context provided about office locations and related research areas does not detract from the correctness.
- **Groundedness:** 0.40 — While the claim about Zijun Yao is fully supported by the context, the answer introduces 'David Johnson' with specific details (office, email, and research interests including Natural Language Understanding) that do not appear anywhere in the retrieved context, constituting a clear hallucination.

### [RS02] Who at KU EECS conducts research in wireless communications or signal processing?
- **Score:** 0.75 | **Difficulty:** research
- **Correctness:** 1.00 — The system answer correctly identifies all three key faculty members from the ground truth (Shannon/S.D. Blunt, Victor Frost, and Joseph Evans) and provides additional relevant faculty members. The core facts align perfectly with the ground truth answer.
- **Groundedness:** 0.35 — While the Communication Systems faculty listed in the answer are directly supported by the context, the Signal Processing faculty (Carlton Leuschen, David Petr, Suzanne Shontz) and the claim that Blunt, Hui, Perrins, and Prescott appear in both clusters are not supported by the provided context — the Signal Processing cluster overview is truncated and contains no faculty names. These specific names and the cross-cluster membership claims are hallucinated.

### [RS03] What graduate-level EECS courses are available that focus on theoretical computer science or formal methods?
- **Score:** 0.31 | **Difficulty:** research
- **Correctness:** 0.00 — The system refused to answer the question, claiming the context only covers graduate funding information. The ground truth indicates there are EECS courses at 700+ level with theory/formal topics that should be referenced, so the system's complete refusal to provide any course information results in a score of 0.0.
- **Groundedness:** 1.00 — The answer correctly states that the context only contains graduate funding information and does not include course listings. The suggestions provided (checking the KU website, course catalog, or contacting an advisor) are reasonable deflections rather than fabricated claims from the context.

### [RS04] Which KU EECS professor has the most interdisciplinary research profile, spanning both engineering systems and artificial intelligence?
- **Score:** 0.47 | **Difficulty:** research
- **Correctness:** 0.75 — The system answer correctly identifies Michael Branicky as the strongest match for the most interdisciplinary research profile spanning engineering systems and AI, which aligns with the ground truth. However, it muddies the conclusion by suggesting David Johnson might be 'arguably the most cross-disciplinary,' introducing ambiguity not present in the ground truth.
- **Groundedness:** 0.05 — The answer heavily hallucinates details about Professor Michael Branicky (office, research areas, Dean role) and Professor David Johnson's research (robotics, human-robot interaction, natural language, linguistics) — none of these specific claims appear in the retrieved context. The context only shows David Johnson's name and partial title, and contains no information about Branicky at all.

### [RS05] If a graduate student wants to specialize in cybersecurity at KU, what is the recommended coursework path and who are the key faculty to work with?
- **Score:** 0.36 | **Difficulty:** research
- **Correctness:** 0.50 — The system answer correctly identifies the key faculty (Bardas, Davidson, Hoque, Kulkarni) mentioned in the ground truth, plus additional faculty not in the ground truth. However, it fails to provide the recommended coursework path (EECS 563 and 700-level security electives), claiming this information is not in the provided context, which results in a partial score.
- **Groundedness:** 0.20 — The answer heavily hallucinates faculty members not present in the retrieved context (Bo Luo, Fengjun Li, Drew J. Davidson, Tamzidul Hoque, Sumaiya Shomaji, Prasad Kulkarni, Hossein Saiedian, Debabrata Dey, Lidan Fan) and fabricates contact details for them. Only Alexandru Bardas, Jose Pius Nedumkallel, and Karthik Srinivasan are mentioned in the context, and the answer correctly notes the absence of coursework information.

### [RS06] Which EECS faculty member is a subject matter expert for the White House Office of Science and Technology Policy?
- **Score:** 0.40 | **Difficulty:** research
- **Correctness:** 0.00 — The system refused to provide an answer, claiming the information was not in the context, rather than identifying Shannon Blunt as the RF spectrum R&D SME for OSTP.
- **Groundedness:** 1.00 — The answer correctly states that the context does not contain information about White House advisory roles, and the contact details provided (eecs-info@ku.edu and 785-864-4620) are directly present in the retrieved context.

### [RS07] What distinguishes EECS 658 from EECS 836? Are they the same machine learning course or different?
- **Score:** 0.81 | **Difficulty:** research
- **Correctness:** 0.75 — The system answer correctly identifies that EECS 658 and EECS 836 are different courses at different curriculum levels, with EECS 658 being undergraduate/introductory and EECS 836 being graduate-level, which matches the ground truth. However, the system adds some additional details about application focus and prerequisites that go beyond the ground truth, and slightly mischaracterizes EECS 836 as 'more accessible' when it's actually the graduate-level course.
- **Groundedness:** 0.85 — Most claims are directly supported by the context: the software engineering focus of EECS 658, the Tom Mitchell quote for EECS 836, the prerequisites for both courses, and the credit hours are all grounded. However, the characterization of EECS 836 as 'theoretically broad' versus EECS 658 as more 'application/engineering-focused' involves some interpretive inference beyond what the truncated descriptions explicitly state, and the claim about 'different levels of the curriculum' is not explicitly supported.

### [RS08] Which KU EECS researchers would be most relevant to collaborate with on a project involving hardware security or IoT device security?
- **Score:** 0.69 | **Difficulty:** research
- **Correctness:** 0.75 — The system answer correctly identifies Tamzidul Hoque and Michael Branicky, who are both in the ground truth. However, it omits Alexandru Bardas (enterprise security) who is in the ground truth, while adding several additional researchers not mentioned in the ground truth (Shomaji, Alexander, Fengjun Li, Davidson). The core ground truth researchers are mostly covered, but one is missing and the answer includes extra information beyond the ground truth scope.
- **Groundedness:** 0.25 — Only two professors mentioned in the answer are supported by the retrieved context: Tamzidul Hoque (hardware security) and Fengjun Li (IoT security). The other four professors mentioned — Sumaiya Shomaji, Perry Alexander, Michael Branicky, and Drew J. Davidson — along with their research interests, office locations, and email addresses, are completely fabricated and not present in the provided context.

### [RS09] Which EECS courses overlap conceptually with mathematics in areas like optimization or probability theory?
- **Score:** 0.76 | **Difficulty:** research
- **Correctness:** 0.50 — The system answer correctly identifies EECS 658 and EECS 461 as courses overlapping with mathematics (matching the ground truth), but misidentifies EECS 868 instead of EECS 836 for optimization in ML, and adds many extra courses not in the ground truth. The core courses from the ground truth are partially covered but with one key error (EECS 836 vs EECS 868).
- **Groundedness:** 0.85 — 

### [RS10] How many EECS faculty at KU have research interests related to robotics or autonomous systems?
- **Score:** 0.69 | **Difficulty:** research
- **Correctness:** 0.75 — The system answer correctly identifies the two faculty members mentioned in the ground truth (Arvin Agah and Michael Branicky) and goes further to identify additional faculty, which is consistent with the ground truth's 'at least 3' framing. The answer is more complete than the ground truth requires, and the core facts align well.
- **Groundedness:** 0.25 — Only 2 of the 6 faculty members listed (David Johnson and Michael Branicky) are directly supported by the retrieved context with explicit robotics-related research interests. The other 4 faculty members (Arvin Agah, Huazhen Fang, Brian McClendon, and Shian Wang) and their research interests are not present in the provided context, making the majority of the answer hallucinated.

## Professor Panel Notes

### What BabyJay does well
- Direct single-field lookups (email, credits, tuition rates)
- Domain-specific routing to correct retriever
- Cross-encoder reranking improves top-result quality
- Hybrid BM25 + vector search for course retrieval

### Known limitations
- Multi-hop prerequisite chain traversal (requires chaining multiple lookups)
- Arithmetic computation (tuition × hours requires calculator, not retrieval)
- Intersection queries across multiple research interest fields
- Date arithmetic (computing number of weeks between two dates)

### Architecture
- Embeddings: `text-embedding-3-large` (OpenAI)
- Vector DB: ChromaDB with persistent storage
- Retrieval: BM25 + vector hybrid with Reciprocal Rank Fusion
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no API)
- LLM: `claude-sonnet-4-6` for answer generation
