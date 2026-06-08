# Mini-Assignment 3 - Architecture

This document captures the design decisions for Mini-Assignment 3, the
capstone deliverable that wraps the MA1-domain-adapted and MA2-aligned
SmolLM2-360M model into a working system. It is a development document,
not the academic report; report-ready content is extracted from here.

The companion implementation reports for the prior stages are
[`ma1_implementation_report.md`](ma1_implementation_report.md) and
[`ma2_implementation_report.md`](ma2_implementation_report.md). Reviewers
read all three together; the MA3 contribution is meant to be coherent
with the arc rather than a new project.

## 1. Goal and shape of the system

The MA3 brief asks for a working system that integrates at least two
additional course components on top of the MA1 + MA2 model. The
application chosen is a **recruiter work-product bundle generator**: given
a free-form recruiter request (e.g. *"We need a Django backend engineer
on AWS, mid-level, fully remote, Lisbon"*) the system produces a
structured Markdown bundle containing four sections - a Job Posting, a
Compensation band, a 30/60/90-day Onboarding Plan, and a Welcome Kit. The
bundle is what an HR practitioner would assemble manually for a new
opening; automating it is a real, narrow use case for an aligned domain
model.

MA3 ships two parallel implementations of this bundle generator under a
shared input/output contract. The deliverable is not one production system
but the **comparison between the two** - an edge-deployable specialist
built on the MA2 model, and an infrastructure-rich orchestrator built on
a larger general-purpose model with GraphRAG. The comparison itself is
the engineering contribution: it isolates what consumer-grade alignment
buys against what production-grade serving buys, on the same task, with
the same evaluation harness.

### 1.1 Why two architectures rather than one

The MA3 brief permits switching the production core model if the
MA2-aligned model is unsuitable. Both choices were considered.

A single-architecture MA3 centred on DPO-b01 forces the bundle generator
to rely on the small model for tasks - retrieval-augmented generation,
tool calling, instruction-following on a novel prompt schema - that
empirical testing showed it cannot reliably perform (Section 2.1). A
single-architecture MA3 centred on Gemma 4 reduces the MA1+MA2 work to a
baseline that has no place in the production pipeline, which weakens the
three-assignment arc the brief explicitly rewards.

The dual-architecture design preserves both signals. DPO-b01 is the
production core for Architecture A and the structural specialist its
training was designed for. Gemma 4 is the production core for Architecture
B and the general-purpose orchestrator that handles the parts a
template-locked small model cannot. The 20 evaluation prompts from MA2's
Section 6.1 measure them on the same axes; the operational profile of
each (latency, VRAM, artefact size, deployment complexity) is measured
alongside. The MA3 report frames the result as a trade-off, not a
ranking.

## 2. Empirical findings that drove the architecture

Two pre-MA3 experiments fixed the architecture direction.

### 2.1 DPO-b01 is template-locked

A three-variant prompt test ran the DPO-b01 model with three preamble
variants on the same recruiter query, all under the canonical generation
settings from `ma2s62code` (greedy, repetition_penalty=1.3,
max_new_tokens=4096). The variants and their results are reproduced in
[`ma3_prompt_test.md`](ma3_prompt_test.md):

| Variant | Preamble | Sections produced |
|---|---|---:|
| A | Canonical MA2 SFT preamble | 3 of 4 |
| B | Preamble names the 4 sections explicitly | 2 of 4 |
| C | Preamble + post-Request reminder | 0 of 4 (template broken) |

Adding any instruction not present in the SFT preamble degrades the
output. Variant C is the load-bearing finding: a single reminder line
between the `### Request` block and the `### Posting` block moved the
model off the canonical `## Heading` template and into a bullet-list
format it never saw at training time.

The lineage explains it. SmolLM2-360M base
([`ma1_implementation_report.md:107`](ma1_implementation_report.md))
was deliberately the non-Instruct variant; MA1 continued-pretrained on
raw text without any prompt-following supervision; MA2's SFT trained one
fixed template; MA2's DPO ranked candidates that all share the same
template. The model has never been exposed to "follow novel
instructions" as a generalisable capability. Variant C is the predicted
behaviour, not an edge case.

Direct consequence for MA3: any architecture that relies on prepending
retrieved context, function-call schemas, or chain-of-thought scaffolding
to DPO-b01 will not work. Architecture A is therefore designed without
in-context grounding; all retrieval is moved outside the model's prompt.

### 2.2 The Compensation section is structured-lookup, not retrieval

The Stack Overflow Annual Developer Survey 2025
(`data/jobs/stack_overflow/results.csv`, 49,191 respondents, 172 columns)
was verified as the salary backbone of the Compensation section. Working
set with `Country`, `ConvertedCompYearly`, and `DevType` all populated:
23,947 respondents (48.7%). EU-27 coverage 36.8%; Portugal 216
respondents with usable medians (n>=10) on full-stack, back-end,
architect, and front-end role families.

Because the data is structured and discrete, vector retrieval is the
wrong tool. The right shape is a function-callable salary lookup:
`get_salary_band(role_family, country, seniority_band) -> {median_usd,
iqr, n, source}`, backed by a pandas / SQLite query over the survey CSV.
Architecture B exposes this as a tool that Gemma 4 calls; Architecture A
calls the same function directly from its assembly script. The corpus of
PDFs (Section 8) handles the narrative compensation framing only; the
numbers come from the table.

## 3. The shared bundle contract

Both architectures speak the same input/output contract. This is what
makes the comparison apples-to-apples.

### 3.1 Input

A single free-form string, the recruiter request, parsed by an
input-normalisation stage into a structured spec:

```
{
  "role_family": "backend" | "frontend" | "mobile" | "ml" | "data" | ...
  "role_title":  string                  # the recruiter's exact phrasing
  "seniority":   "junior" | "mid" | "senior" | "lead" | null
  "location":    country code or city (free text, normalised to country)
  "work_mode":   "onsite" | "hybrid" | "remote" | null
  "stack_required":  [string, ...]       # extracted tech mentions
  "stack_preferred": [string, ...]       # softer mentions
  "raw_request":    string               # the original input
}
```

The parser is a lightweight regex + dictionary lookup over the entity
glossary built from ESCO + Wikidata + cloud-service catalogs (the
entity-normalisation index, Section 9.2). Architecture A and Architecture
B share the parser.

### 3.2 Output

Markdown document with exactly four `##` sections in this order:

```
# {role_title}

## Job Posting
[Summary / Required Skills / Responsibilities / Requirements]

## Compensation
[Salary band with source attribution, currency, sample size]

## Onboarding Plan (30/60/90)
[Phased plan: Days 0-7, 7-30, 30-60, 60-90]

## Welcome Kit
[Hardware, accounts, documentation, first-day schedule]
```

The `## Job Posting` section preserves the MA2 four-section template
(`Summary`, `Required Skills`, `Responsibilities`, `Requirements`) so
that the MA2 baseline numbers carry across into the MA3 comparison
without renormalisation.

## 4. Architecture A: edge specialist (DPO-b01 core)

The edge architecture treats DPO-b01 as a fast template-shaped generator
embedded in a thin deterministic shell. No external LLM is invoked at
inference time; the entire system fits in roughly 400 MB of artefact
plus a few static tables.

### 4.1 Generator stage

DPO-b01 produces the `## Job Posting` section. The model is loaded from
the merged MA2-SFT base
(`outputs/ma2-360m-sft-merged/`,
[`ma2_implementation_report.md:78`](ma2_implementation_report.md))
plus the DPO-b01 LoRA adapter (`outputs/ma2-360m-dpo-b01/`), then merged
into a single consolidated model and quantised to GGUF for inference.

Quantisation chain (item 6 on the brief's component list, performance
optimisation):

1. Merge the DPO-b01 LoRA into the SFT-merged base. Produces a single
   `safetensors` checkpoint of approximately 720 MB in bf16.
2. Convert to GGUF using `llama.cpp`'s `convert_hf_to_gguf.py`.
3. Quantise to `Q8_0` using `llama-quantize`. Final size approximately
   380 MB. (`Q4_K_M` reduces further to approximately 230 MB; whether to
   use `Q4` or `Q8` is a separate measurement, the default of record is
   `Q8` because at 360M parameters the quality cost of `Q8` is
   negligible.)
4. Register with `agent_server` as a chat model so the same serving
   stack hosts both architectures and the operational comparison is
   apples-to-apples.

Inference settings match `ma2s62code` exactly: greedy decoding,
`repetition_penalty=1.3` (the kwarg whose absence caused the v6
first-pass catastrophe documented in MA2 §11.1), `pad_token_id` synced
to `eos_token_id`, max 4096 new tokens.

### 4.2 Best-of-N section gate

The MA2 evaluation generations established that DPO-b01 emits all four
template sections in 1 of 20 prompts under greedy decoding (the
project-memory note for `ma2s66` / `ma2s71` / `ma2s68` records this
explicitly). To address this without changing the model, Architecture A
generates **N=5** candidates per request with different sampling seeds
(`temperature=0.9`, `top_p=0.95`), checks each for the four `##`
headings, and returns the first 4-of-4 candidate found. If no candidate
hits four sections, the candidate with the highest section count wins.

This is item 4 on the brief's component list, iterative
self-improvement. The cost is 5x the inference time per request:
approximately 15 s per bundle on the RTX 4090 at Q8, against
approximately 3 s for a single greedy generation. Architecture A's
"self-contained / fast" framing survives this cost because Gemma 4
(Architecture B) plus retrieval is slower still.

### 4.3 Deterministic table lookups

`## Compensation` is filled by calling `get_salary_band(role_family,
country, seniority_band)`. The function reads
`data/jobs/stack_overflow/results.csv`, applies the (role_family x
country x seniority) filter, returns `{median_usd, iqr, n, source}`. No
LLM involved. Architecture A wires this into a Markdown formatter that
renders the result with source attribution.

`## Onboarding Plan (30/60/90)` and `## Welcome Kit` come from a
template registry indexed by role family. The templates are stored as
Jinja-rendered Markdown files, one per role family, with placeholders
for role title and stack. Architecture A selects the template by the
parsed `role_family` and substitutes. No LLM, no RAG.

The template registry is small (12 onboarding templates + 6 welcome-kit
templates) and is part of Architecture A's frozen artefact. It is the
hardcoded counterpart of Architecture B's GraphRAG retrieval.

### 4.4 Operational profile

- Total artefact size: approximately 400 MB (380 MB quantised model +
  static tables and templates).
- Inference VRAM: approximately 2 GB at Q8 on the RTX 4090.
- Latency per bundle: approximately 15 s under best-of-5 + table
  lookups.
- External dependencies at inference: none. The system runs offline
  once the artefact is on disk.

This is the deployability end of the trade-off the MA3 report frames in
Section 12.

## 5. Architecture B: orchestrator with GraphRAG (Gemma 4 core)

The orchestrator architecture treats DPO-b01 as a historical baseline
and uses Gemma 4 as the production generator. Gemma 4 plans the bundle,
calls structured tools for the parts that need exact data, and calls
GraphRAG for the parts that need free-text grounding.

### 5.1 Orchestration

Gemma 4 is served by the local `agent_server` stack (see
[`agent_server presets memory note`](../../../.claude/projects/-home-logus-env-iscte-atlm-pro/memory/agent-server-presets.md)
and MA2 §9 for the calling pattern). An `agent_server` preset
`atlm_ma3_orchestrator` carries the bundle prompt: it describes the
four-section contract, lists the available tools, and instructs Gemma 4
to think step by step through which tool calls and retrievals it needs.
That chain-of-thought layer is item 1 on the brief's component list,
advanced reasoning.

The orchestrator plan, expressed as the sequence of tool calls Gemma 4
generates for a typical request:

1. `get_salary_band(role_family, country, seniority)` for the
   `## Compensation` block.
2. `graphrag_search(query="onboarding plan for X developer",
   domains=[jobs_onboard_general, jobs_onboard_{role_family}])` for
   `## Onboarding Plan`.
3. `graphrag_search(query="welcome kit first day", domains=[
   jobs_onboard_welcome])` for `## Welcome Kit`.
4. `graphrag_search(query="job description inclusive {role}",
   domains=[jobs_writing, jobs_inclusive, jobs_ladders])` for the
   Job Posting style and tone.
5. Compose the four sections into the final bundle.

### 5.2 Function-calling tool registry

The tool registry exposes a small set of structured operations Gemma 4
can dispatch via function calling. This is item 3 on the brief's
component list, tool use / agentic behaviour.

| Tool | Backed by | Returns |
|---|---|---|
| `get_salary_band(role_family, country, seniority)` | Stack Overflow Survey 2025 CSV via pandas | `{median_usd, iqr, n, source}` |
| `parse_request(raw_request)` | Regex + ESCO/Wikidata entity dictionary | The structured spec (Section 3.1) |
| `graphrag_search(query, domains)` | noted-rag `POST /search_multi` over the named Domains | List of chunks with `score`, `text`, `source_path`, `kb_id` |

Function calling is run through `agent_server`'s preset machinery
following the same pattern MA2 used for its judge agents (MA2 §9). The
agent receives the tool schemas in its system prompt; tool outputs come
back as structured arguments.

### 5.3 GraphRAG retrieval

The retrieval layer is the existing `noted-rag` + `noted-graph` stack
(see Section 7 for the API surface). Per-Domain
GraphRAG: each topic folder in our corpus (Section 8) is ingested as a
separate Domain with its own ChromaDB vector collection AND its own
extracted graph of entities and relationships. Section 5.3 in the
Knowledge Base Monitor screenshot for `jobs_onboard_backend` showed 318
graph entities and 374 relationships from the first source alone; that
is the relationship-aware grounding signal that distinguishes GraphRAG
from pure vector RAG.

For MA3, the bundle generator calls `POST /search_multi` against a
selected subset of Domains per bundle section. Multi-Domain queries are
single-rerank-batch (`noted-rag/app/rag_service.py` line 338 onward) so
the latency cost of touching three Domains in one search is one embed +
N HNSW lookups + one rerank batch, not N round trips. This is item 2 on
the brief's component list, RAG.

### 5.4 Operational profile

- Total resident artefact: Gemma 4 weights (approximately 5 GB for the
  E4B variant) plus `bge-m3` embedder (approximately 600 MB) plus
  `bge-reranker-v2-m3` (approximately 600 MB) plus the ChromaDB
  collections plus the graph database. Order of magnitude: 10-12 GB.
- Inference VRAM: approximately 10 GB resident (Gemma 4 plus the two
  bge models always resident under `agent_server`'s policy).
- Latency per bundle: estimated 30-45 s on the RTX 4090, including the
  per-section retrievals, the tool calls, and Gemma 4's generation.
- External dependencies at inference: the full `agent_server` stack
  (Docker Compose with `agent-server` and `agent-server-llama-adapter`
  images), `noted-rag`, `noted-graph`, ChromaDB on a writable volume.

This is the quality end of the trade-off.

## 6. Component mapping to the MA3 brief

The brief requires at least two of seven listed components beyond the
MA1 + MA2 work. The MA3 system earns six of them, distributed across
the two architectures and the shared evaluation harness. The honest
mapping:

| Component | Where it lives | Architecture |
|---|---|---|
| Advanced reasoning (CoT) | Gemma 4's bundle-planning preset prompt | B |
| RAG | GraphRAG via noted-rag, per-Domain multi-collection search | B |
| Tool use / agentic | `agent_server` function calling for salary, parsing, retrieval | B |
| Iterative self-improvement | Best-of-N section gate on DPO-b01 | A |
| LLM-as-judge evaluation | Qwen3.5-9B (or similar non-Gemma family) pairwise judging | Shared |
| Performance optimisation | Q8 (or Q4) GGUF quantisation of DPO-b01, served via llama.cpp | A |

Further alignment or post-training is on the brief's list but not used
here; doing another DPO round would change MA2's reported numbers and
adds disproportionate cost for marginal gain.

## 7. Reused infrastructure (noted-rag, noted-graph, agent_server)

MA3 does not reinvent infrastructure. The retrieval and serving layers
already exist in the user's stack and are reused as production
components. Three services participate.

### 7.1 noted-rag (FastAPI sidecar)

Endpoints exercised by the MA3 bundle generator:

| Endpoint | Purpose |
|---|---|
| `POST /search` | Single-collection dense retrieve + cross-encoder rerank to top-k |
| `POST /search_multi` | Multi-collection retrieval with one merged rerank batch |
| `POST /embed` | `bge-m3` vectors via the llama-server router |
| `GET /collections` | Sanity check on the post-ingestion collection state |

Retrieval semantics: ChromaDB HNSW dense top-K (default `DENSE_TOP_K`),
then `bge-reranker-v2-m3` cross-encoder reranks to final top-k. A
configurable `RERANK_MIN_SCORE` floor returns empty rather than noise
when no candidate exceeds the threshold. For the recruiter prompts the
floor will be set lower than the default (the noted-rag code already
exposes `rerank_min_score` as a per-request override for exactly this
case, see `app/main.py` line 143).

### 7.2 noted-graph (Docling ingestion + graph extraction)

Per-Domain ingestion runs Docling over the corpus PDFs, chunks the
extracted text, ships chunks to noted-rag's `POST /upsert_chunks`, and
extracts entities and relationships into a graph database. The Knowledge
Base Monitor exposes the live state of both the vector and graph
ingestion (see screenshots; for `jobs_onboard_backend` at 15.6% complete:
75 vector chunks indexed, 318 graph entities, 374 graph relationships
extracted, EXTRACTING phase ongoing).

MA3 does not modify noted-graph. It uses the Domains it produces.

### 7.3 agent_server

The OpenAI-compatible LLM serving stack from MA2 (MA2 §9). MA3 adds two
new agent presets:

- `atlm_ma3_orchestrator` - Gemma 4 with the bundle prompt and the tool
  registry described in Section 5.2.
- `atlm_ma3_judge` - Qwen3.5-9B-class with the pairwise rubric described
  in Section 10.

Calling pattern is unchanged: `model: "<agent_name>"` in the chat
completions payload, no inline system prompt, the rubric and tool
schemas resolved server-side. The single-resident-chat-model invariant
(MA2 §9.2) means orchestrator and judge cannot serve concurrently; a
60-second `switch_active_model` hold mediates between them, matching the
MA2 pattern.

## 8. Knowledge corpus

### 8.1 Assembly methodology

No single public dataset covers the topic surface the bundle requires.
Engineering onboarding is documented across scattered company
handbooks, HR-software template galleries, university HR offices,
academic case studies, and government / standards bodies. The corpus
was assembled by targeted retrieval and per-Domain curation rather
than by ingesting any pre-packaged corpus end to end.

**The fetching pipeline**, applied iteratively for each of the 19
Domains:

1. *Targeted search.* For each Domain, one or more `WebSearch` queries
   with a `filetype:pdf` constraint. Queries followed three patterns:
   a topic-anchored pattern (e.g. `"developer onboarding" filetype:pdf
   engineering handbook`), a role-family pattern (e.g.
   `"backend developer" onboarding plan filetype:pdf`), and a
   support-topic pattern (e.g. `"first day" new hire checklist welcome
   kit engineering filetype:pdf`). Each search returned 5-10 candidate
   URLs.
2. *Source triage from the search-result metadata.* Each candidate was
   classified before download:
   - Authoritative direct sources (Google SRE Workbook, OWASP
     publications, university HR offices, government open data,
     vendor documentation): kept.
   - Engineering handbooks from real companies (GitLab handbook,
     Etsy / Square / Block / Monzo engineering ladders): kept;
     these supplied per-role grounding nothing else could.
   - HR-software template marketing PDFs (Workable, BambooHR, iCIMS,
     ExecOnline, FlareHR): kept when content was substantive, skipped
     when essentially marketing brochures.
   - Pirated book scans on file-hosting domains (ebooks.karbust.me,
     pdfcoffee.com, soclibrary.futa.edu.ng, course-hero scrapes,
     z-library mirrors): skipped on copyright grounds even when the
     content would have fit.
3. *Download with explicit failure semantics.* `curl -fsSL` per
   candidate. The `-f` flag makes HTTP error responses (404, 403,
   redirects to auth pages) fail the download instead of writing an
   HTML body to a `.pdf` file. Targets written to topic-specific
   folders under `data/rag_corpus/` using absolute paths after one
   round of `cd`-relative-path mistakes scattered files across the
   filesystem.
4. *Format validation.* `file -b` for the magic-byte check and
   `pdfinfo` for the page count, run on every downloaded artefact.
   Files that returned non-PDF content (one early candidate
   `opm_getting_on_board.pdf` from a federal-data URL turned out to be
   an HTML landing page) were deleted and not replaced. Files that
   downloaded as empty (`mend_ciso_appsec_innovation.pdf` returned a
   0-byte file - the Mend.io endpoint blocks `curl` user agents) were
   deleted.
5. *Gap analysis and a second round of searches.* After the first
   pass, four folders held only two PDFs each (backend, frontend,
   mobile, security). A second round of more specific searches
   targeted those gaps with queries like `Django OR "Spring Boot"
   OR "Node.js" engineering handbook filetype:pdf` and `OWASP
   application security developer guide filetype:pdf`. The second
   pass brought all role folders to at least 3 PDFs and most to 5-6.
6. *Domain expansion.* Once the onboarding folders were filled, six
   additional Domains were added for non-onboarding content that the
   bundle needs (career ladders, inclusive language, engineering
   culture, compensation writing, remote/async, interview process)
   and a final round of searches populated those. The full corpus
   went from 49 PDFs after the first round to 100 PDFs after the
   final round.

**Honest failures**, documented because they are reproducible:

- *Sintef SE+ 2017 paper*: host `sintef.brage.unit.no` did not
  resolve from the build environment. Document is real but
  unreachable from this network.
- *MITRE Systems Engineering Guide*: 403 even with `User-Agent`
  spoofing; the MITRE web infrastructure rejects automated
  downloads. The document is publicly published.
- *Mend.io CISO Guide to AppSec*: anti-scraping returns a 0-byte
  response.
- *core.ac.uk "Cyber Onboarding is Broken"*: 404 at the time of
  fetching.
- *Sourcegraph Developer Onboarding e-book*: HubSpot-served PDF whose
  metadata reports `pdfinfo` page count of 0 because of a libpoppler
  quirk; `pdftotext` extracts the content cleanly. Document retained
  in the corpus.

**Discipline rule applied throughout** (from
[`feedback-check-the-data.md`](../../../.claude/projects/-home-logus-env-iscte-atlm-pro/memory/feedback-check-the-data.md)
in the session memory): validate field-population against actual
content, not against schema descriptions or LLM-suggested dataset
names. An earlier candidate dataset (`data/jobs/eu-tech-jobs/`) was
inspected before commitment and rejected when its company-stack
columns (`oss_signal`, `github_org`, `top_repo_stars`,
`primary_language`) turned out to be 0 - 2% populated, even though
the schema implied rich coverage. The same discipline was applied to
each PDF candidate: searching for and downloading is cheap, ingesting
something useless is expensive.

**Initial framing discarded.** The first round of analysis assumed a
vector-RAG corpus over job postings would be the right shape. Two
findings reframed it: (i) job-posting-style corpora overlap the model's
training distribution and would not add new information, and (ii) the
LinkedIn corpus used as the MA1 out-of-distribution probe would
contaminate that probe if used for retrieval. The corpus was
re-scoped to information neither training stage saw: onboarding,
welcome-kit, career ladders, compensation writing, inclusive language,
engineering culture, remote/async practices, and interview process. The
vector / structured split (Section 2.2) emerged from the same reframe.

### 8.2 Corpus inventory

The corpus is at `data/rag_corpus/`, organised as 19 topic folders
mapping one-to-one onto noted Domains. Final state at the time of
writing:

| Folder | Domain slug (proposed) | PDFs | Pages | Notes |
|---|---|---:|---:|---|
| `onboarding_general` | `jobs_onboard_general` | 12 | 224 | Sourcegraph dev-onboarding, Bridgespan, FlareHR, Berkeley remote, Terminal remote |
| `onboarding_backend` | `jobs_onboard_backend` | 5 | 290 | Spring Boot reference, Apigee API design, Mosh and roadmap.sh roadmaps |
| `onboarding_frontend` | `jobs_onboard_frontend` | 5 | 776 | Frontend handbook, TypeScript Deep Dive, professional FE architecture |
| `onboarding_mobile` | `jobs_onboard_mobile` | 6 | 2,083 | Apple Swift docs, Kotlin elements, mobile app dev textbooks |
| `onboarding_ml_ai` | `jobs_onboard_ml` | 4 | 74 | Google + AWS ML cert guides, Mosh ML roadmap, Zinkevich Rules of ML |
| `onboarding_data_eng` | `jobs_onboard_data` | 3 | 187 | ThoughtWorks, StreamSets, 3CloudSolutions data engineering |
| `onboarding_devops_cloud` | `jobs_onboard_devops` | 4 | 649 | Google SRE Workbook (canonical), New Relic SRE, DevOps Institute |
| `onboarding_qa` | `jobs_onboard_qa` | 3 | 81 | Global App Testing QA handbook, roadmap.sh QA, CodeSignal eval |
| `onboarding_security` | `jobs_onboard_security` | 6 | 705 | OWASP Developer Guide, OWASP ASVS, OWASP Testing Guide, CMU SEI |
| `onboarding_embedded` | `jobs_onboard_embedded` | 3 | 608 | UC Berkeley embedded textbook (canonical), firmware curriculum |
| `onboarding_architect` | `jobs_onboard_arch` | 3 | 341 | Will Larson Staff Engineer (canonical), agile team onboarding |
| `welcome_kit_day1` | `jobs_onboard_welcome` | 5 | 47 | LinkedIn Onboarding in a Box, Adobe checklist, DOI federal checklist |
| `job_posting_writing` | `jobs_writing` | 9 | 126 | TechPoint inclusive JD playbook, CIPD UK, Apprenticeship.gov skills-first |
| `career_ladders_leveling` | `jobs_ladders` | 5 | 92 | Square, Block, Monzo, Etsy, Agile Lab engineering ladders |
| `inclusive_language_dei` | `jobs_inclusive` | 6 | 103 | Illinois, Inclusive Employers Canada, Montgomery County, Berkeley Haas |
| `engineering_culture` | `jobs_culture` | 4 | 88 | VIRTA, Arcadia engineering strategy + values, culture index |
| `compensation_benefits_writing` | `jobs_compensation` | 8 | 182 | WorldAtWork total rewards, SHRM, Google pay transparency, NBER |
| `remote_async_work` | `jobs_remote` | 3 | 114 | Async-First Playbook, TechSmith experiment, GitLab Remote Playbook |
| `tech_interview_process` | `jobs_interview` | 6 | 55 | Villanova, CMU, Princeton tech interview guides, Parnin "Debugging Hiring" |

Totals: 100 PDFs, ~6,825 pages, ~360 MB on disk.

Sourcing rules applied during corpus assembly:

- Authoritative sources preferred (Google SRE Workbook, OWASP guides,
  UC Berkeley textbook, Apple Swift docs, Will Larson's Staff Engineer)
  over generic HR-software marketing content.
- Pirated book scans and copyright-questionable redistributions skipped
  (clearly noted in `documents/development/rag_dataset_analysis.md` and
  in the curation pass).
- File-type validated (PDF magic bytes, page count) at download time;
  one document (`opm_getting_on_board.pdf` originally) was rejected
  because the server returned HTML masquerading as PDF.
- Cross-industry breadth (LinkedIn cross-industry onboarding,
  university HR checklists) included alongside engineering-specific
  guides so the GraphRAG can ground both technical and procedural
  recruiter requests.

## 9. The variant test as a methodological artefact

The Section 2.1 variant test is preserved as a reproducibility artefact
at `documents/development/ma3_prompt_test.md`. It includes the three
exact prompts sent to DPO-b01, the three full generations, and the
section-count table. It is part of the MA3 report's critical-discussion
section, not a hidden internal note.

The methodological value is that it falsifies an architectural option
before any infrastructure was committed to it. The first three iterations
of the MA3 design assumed prompt-injected RAG would work; the variant
test ruled that out. Architecture A's design (no in-context grounding;
all retrieval external; structure repair via best-of-N) and Architecture
B's design (orchestrator does the prompt-aware work; the small model
stays in its trained shape) both descend from the variant-test result.

This is the kind of pivot the brief explicitly rewards under "Critical
discussion - what worked, what didn't, where the system fails, what
you'd build differently with more time".

## 10. Evaluation design

The evaluation reuses the MA2 protocol verbatim where possible. New
work is the bundle-specific rubric and the operational measurements.

### 10.1 Prompt set

The 20 evaluation prompts from MA2 §6.1, frozen at
`data/processed/ma2/eval_prompts.jsonl`. Ten `ind-*` in-distribution
(written in the style of training data, never seen by either model
during training), ten `ood-*` out-of-distribution (junior bootcamp,
principal staff, niche stacks, freelance technical writer, soft-skills
roles, remote-async). Reusing MA2's set lets the report carry the MA2
numbers across into the MA3 comparison without renormalisation.

### 10.2 Baselines

Four baselines, the first two mandated by the brief, the latter two
isolating component contributions.

- (a) Base SmolLM2-360M (no MA1, no MA2 adapters, no system). Brief
  mandate. Naturally weakest; produces text completion not structured
  output.
- (b) DPO-b01 alone with no system components. Brief mandate. Same as
  the MA2 §6 evaluation; the numbers carry across.
- (c) Gemma 4 alone via `agent_server`, with the structured bundle
  prompt but no RAG and no tool calls. Isolates what Gemma 4 brings
  in by raw capability versus what RAG and tools add.
- (d) Architecture A (Q8 DPO-b01 + best-of-N + deterministic
  tables/templates) and Architecture B (Gemma 4 + GraphRAG + tools) as
  the two production systems under test.

The (c) vs (d-B) comparison answers "what does the GraphRAG and tool
layer add over Gemma 4 alone?" - the contribution of MA3 Architecture B
beyond raw scale. The (b) vs (d-A) comparison answers "what does the
edge wrapping add over the bare MA2 model?" - the contribution of MA3
Architecture A.

### 10.3 LLM-as-judge

A larger non-Gemma, non-Granite judge: Qwen3.5-9B is the first choice
because it has the parameter budget the MA2 Granite ceiling lacked
(documented in MA2 §12.3 as the close-pair discrimination problem) and
sits in a different model family from both production cores. Family
diversity is the same cross-judge principle MA2 used to motivate
Configuration C.

Pairwise rubric, multi-axis:

- Entity correctness. Does every named technology in the output exist
  and use its canonical name? Direct attack on the MA2 EKS / KMS /
  PhpExpress / Unity-for-Unreal failures the MA2 reports document.
- Section completeness. Are all four bundle sections present and
  non-empty? Trivially deterministic to score, but included on the
  judge axis for the cases where headings are correct but content is
  empty.
- Role-stack appropriateness. Does the stack named in the output match
  the role the prompt asked about? Catches the b01 cross-stack drift
  (.NET in a Python role, Unity in an Unreal role, OOP for Elixir).
- Prose quality. Generic HR-boilerplate vs. concrete content.
- Bundle coherence. Do the four sections refer to the same role
  consistently? Specific to MA3 (not measurable on MA2's single-section
  outputs).

Each pair scored AB-BA order-swap per MA2 §6.4 protocol. Only
consistent verdicts are counted as signal; AB-BA disagreement is
reported as inconsistency rate and is itself a diagnostic. Same
protocol Granite carried in MA2; same protocol Qwen3.5-9B carries here.

### 10.4 Operational measurements

For each baseline and the two production architectures, recorded per
run on the RTX 4090:

- VRAM resident during inference (read from `nvidia-smi`).
- Wall-clock latency per bundle (median over the 20 prompts).
- Total artefact size on disk.
- External dependency count (services that must be running for a single
  request to succeed).
- Lines of orchestration code needed to run an end-to-end request.

These numbers populate the Section 12 trade-off table.

### 10.5 Deterministic checks

Where the judge axis can be replaced by a deterministic boolean, it is:

- Entity correctness: per prompt, the set of entities mentioned is
  extracted; each is checked against the canonical glossary; the count
  of "canonical form present in output" vs. "invented form present in
  output" is the deterministic metric. Judge scoring augments rather
  than replaces this.
- Section completeness: per output, count `## ` headings matching the
  four-section contract. Deterministic.
- These are reported alongside the judge results so a reviewer can see
  the judge's verdict against the deterministic ground truth where it
  exists.

## 11. Implementation phasing

Estimated work to take the architecture from this document to running
results, on the assumption that the corpus ingestion (Section 8) is
already underway in parallel.

| Phase | Work | Approx. effort |
|---|---|---|
| 1 | Shared contract layer (`src/ma3/bundle.py`): bundle dataclass, Markdown renderer, input parser, salary-table tool, hand-written onboarding/welcome-kit template registry for Architecture A | 1 day |
| 2 | Quantise DPO-b01 to Q8 GGUF; register with `agent_server`; wire Architecture A end-to-end (best-of-N + section gate + table lookups + assembly) | 1 day |
| 3 | `atlm_ma3_orchestrator` agent preset with the bundle prompt and tool registry; wire Architecture B end-to-end (Gemma 4 + tool dispatch + multi-Domain GraphRAG retrieval + assembly) | 2 days |
| 4 | `atlm_ma3_judge` agent preset; run all four baselines + the two architectures on the 20-prompt set; collect generations | 1 day |
| 5 | Run the pairwise judging passes (AB-BA order-swap, all relevant comparisons); record consistent verdicts and inconsistency rates; collect operational measurements; produce the results tables and qualitative side-by-side examples | 1-2 days |

Total: approximately 6-7 days of focused work after the corpus ingestion
completes. All on infrastructure that exists.

## 12. Trade-off framing

The MA3 result is not "which architecture wins" but "which architecture
fits which deployment context". The expected shape of the trade-off,
without prejudging the numbers:

| Axis | Architecture A (DPO-b01 + tables) | Architecture B (Gemma 4 + GraphRAG) |
|---|---|---|
| Total trainable on a consumer 4090 | Yes - MA1 + MA2 took roughly 3 hours of training end to end | No - Gemma 4 cannot be LoRA-fine-tuned on a 4090 |
| Inference VRAM | ~2 GB | ~10 GB |
| Wall-clock latency | ~15 s/bundle | ~30-45 s/bundle |
| Total artefact size | ~400 MB | ~10-12 GB |
| External deps at inference | None | `agent_server` + `noted-rag` + `noted-graph` + ChromaDB |
| Quality (entity correctness, section completeness, role-stack match, bundle coherence) | Expected lower - the MA2 §12 limits transfer over | Expected higher - Gemma 4 + GraphRAG addresses the four documented failure classes |
| Reproducibility on a student laptop | Trainable end to end on a 16 GB consumer GPU | Inference only; orchestration training out of reach |

The report uses this table to frame the deployment decision the system
forces: *if infrastructure is a constraint, the MA1+MA2 specialist is
the only deployable option, and its quality limits are documented and
known. If infrastructure is not a constraint, the larger orchestrator
with GraphRAG is materially better but cannot be reproduced end to end
on consumer hardware.*

## 13. Arc reflection (preview, finalised after evaluation)

The brief explicitly rewards reflection on whether earlier stages got
washed out by later ones. The variant test (Section 2.1) and the
architecture decision (Section 1.1) give the report a concrete answer
before any results land:

- MA1's continued pretraining produced domain fluency. Both
  architectures consume it: Architecture A directly (b01 = MA2 LoRA on
  MA1 base), Architecture B indirectly (RAG retrieval over the same
  domain corpus). The domain knowledge is not washed out; it shows up
  in the b01 baseline numbers and in the Architecture A production
  numbers.
- MA2's SFT and DPO produced template-shaped generation under the
  alignment signal from Nemotron. Architecture A's quality numbers are
  the direct measurement of what alignment produced. Architecture B
  does not use the aligned weights; Gemma 4 does the generation. The
  alignment work is therefore measurable as Architecture A's quality
  floor, and the difference between A and B quantifies what alignment
  did not buy.
- MA3's contribution is the dual-architecture comparison itself.
  Neither architecture is "the system"; the trade-off they bound is.

This framing makes the brief's "honest reporting of failures and
limitations" requirement structural: the limits are not a sad
appendix, they are the contribution. The MA2 §12 limitations
(Granite judge ceiling, DPO coverage bound, small-content
hallucinations, 360M scale ceiling) become quantitative axes in the
Section 10 evaluation rather than disclaimers.

## 14. Open items at time of writing

Pending corpus ingestion completion:

- Per-Domain GraphRAG retrieval calibration: tune `rerank_min_score`
  per Domain on the 20 evaluation prompts to balance recall against
  noise.
- Domain selection heuristic per bundle section: choose between
  hardcoded mapping (Section 5.1) and a small LLM-as-router upstream of
  the multi-Domain search.

Pending architecture wiring:

- Concrete latency numbers for both architectures under the actual
  hardware.
- Q8 vs Q4 quantisation comparison for DPO-b01 (whether the size win of
  Q4 costs measurable quality at 360M parameters).
- Whether to include a fifth bundle section (`## Required Stack
  Reference` listing the canonical names for technologies mentioned)
  as an explicit Architecture B differentiator.

Pending evaluation:

- Final judge selection (Qwen3.5-9B is the proposed choice; Nemotron is
  the fallback if Qwen3.5-9B's calibration on this task fails the way
  the MA2 calibration battery surfaced Granite's close-pair ceiling).
- The mandatory baselines (a) and (b) of Section 10.2 have outputs
  from the MA2 §6 runs already on disk; only (c) and (d) need to be
  produced.

These are tracked in `documents/development/next-steps.md` (the
session memory) rather than here so that the architecture document
remains stable as those items are resolved.
