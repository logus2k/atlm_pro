# RAG dataset analysis for MA3

Ranked review of the dataset suggestions collected in
`documents/rag_dataset_suggestions.md` (responses from Gemini, Qwen, ChatGPT,
Grok, Claude, CoPilot, DeepSeek, Manus, Z.ai, Mistral). Ordered by:

1. **Practical value** — how directly the dataset addresses one of the three
   MA2 weaknesses (hallucinated niche tech specifics; templated outputs; no
   real-world grounding for company stacks / salary / role detail).
2. **Required effort** — preprocessing, language filtering, schema cleaning,
   embedding/indexing.
3. **Feasibility within the MA3 timeframe** — MA3 is the final deliverable;
   anything that needs a custom scraper, petabyte-scale ETL, or an opaque
   commercial license drops in the ranking.

## Framing decisions that shape the ranking

These are not in the criteria list directly, but they decide where datasets
land in the tiers.

**The model has already seen ~12K Djinni postings during SFT.** Two of the
nine LLMs (ChatGPT and Claude) raise this point explicitly: a retrieval
corpus that is *mostly more job postings* will reinforce the templated
outputs the model already produces rather than ground it in new information.
Postings have a role in the corpus, but they should not be the centre of
gravity. Taxonomies and entity glossaries are where the biggest hallucination
fix lives.

**Two retrieval roles, two indexes.** Claude states the architectural point
explicitly: an entity/normalisation index (exact-match retrieval against
taxonomies, service catalogs, glossaries — used to constrain technical
vocabulary at generation time) is structurally different from a similarity
index (nearest-neighbour retrieval on postings — used for phrasing
diversity). Mixing them into one vector store hurts both. The ranking below
groups recommendations along this split.

**For a 360M model, retrieval quality beats quantity.** A small, curated
corpus of O*NET tech-skills + ESCO + Wikidata glosses + 5-10K real postings
will outperform a million noisy postings, because the model already knows
what postings look like; what it lacks is correct grounding.

**The discipline rule from the eu-tech-jobs incident** (see
[[feedback-check-the-data]] and the project-overview): when an LLM
claims a dataset has rich structured fields (company tech stacks,
salaries, skills, etc.), the population rate of those fields must be
verified against actual records before the dataset earns a high rank. Schema
claims and dataset-card claims are not evidence; only opening the file
counts.

## MA2 weaknesses being targeted

For reference throughout the ranking:

- **W1 — Hallucinated niche technical specifics.** The model invents
  plausible-sounding cloud service names, framework variants, and acronym
  expansions. Strongest example: "EKS" expanded to "Eventual Consistency
  Cluster" instead of "Elastic Kubernetes Service" in MA2 evaluation
  generations.
- **W2 — Templated outputs.** Repeated sentence stems and section openings
  across postings; SFT-driven convergence on a narrow phrasing distribution.
- **W3 — No real-world grounding for company stacks, salary, role details.**
  The model has no signal for what a real Series-B fintech stack actually
  looks like, what a backend role in Lisbon actually pays, or what a senior
  vs. mid-level posting actually differs on.

---

## Tier 1 — High practical value, low effort, fits in MA3 easily

These are clean, open, well-documented, structured sources that directly
attack W1 and provide a stable backbone for the retrieval corpus.

### 1.1 ESCO (European Skills, Competences, Qualifications and Occupations)

- **Source:** European Commission, CC-BY 4.0, fully open. Downloadable as
  CSV, JSON-LD, RDF. Stable URLs, versioned releases.
- **What it injects:** ~13,900 skill/competence concepts, ~3,000 occupations,
  occupation-to-skill mappings, definitions, multilingual labels (including
  Portuguese).
- **Weaknesses addressed:** **W1** (skill normalisation, canonical names)
  and **W3** (responsibility phrasing grounded in the EU taxonomy rather
  than invented). The EU framing is a fit for the Djinni corpus geography.
- **Effort:** Low. Clean CSV; minimal cleaning. A few hours to load, chunk
  per skill/occupation node, embed.
- **Feasibility for MA3:** High. This is the kind of dataset a thesis-scale
  RAG can integrate in a day.
- **Caveats:** Occupation-level, abstract — does not give you product names
  like "EKS". Updated annually, so cutting-edge tools lag.

### 1.2 O*NET (US Department of Labor)

- **Source:** onetcenter.org, CC-BY 4.0. The two files of interest are
  "Technology Skills" and "Tools Used", which map canonical software/platform
  names to each occupation.
- **What it injects:** Task statements per occupation (raw material for the
  Responsibilities section), plus a vetted vocabulary of real tools and
  platforms tied to each role.
- **Weaknesses addressed:** **W1** (Technology Skills file provides a
  vetted set of real software names) and **W3** (occupation-level task
  statements ground the Responsibilities section).
- **Effort:** Low. Public-domain SQL/CSV; straightforward.
- **Feasibility for MA3:** High.
- **Caveats:** US labor market, English-only, SOC-occupation taxonomy.
  "Software Developers" is one code, so it lacks the resolution to
  distinguish "Django backend engineer" from "Spring Boot backend engineer".
  Quarterly updates, conservative on new tools. Pair with a tech-entity
  glossary (Tier 1.3) for product-level resolution.

ESCO and O*NET are partially complementary: ESCO is EU-flavoured and has
multilingual coverage; O*NET has a richer technology-skills file. **Both
together is the right call**, not "pick one". This was consensus across the
LLMs that mentioned either (7/9 named O*NET; 4/9 named ESCO).

### 1.3 Wikidata tech-entity glossary (curated subset)

- **Source:** Wikidata, CC0 (no constraints).
- **What it injects:** Canonical expansions and one-line descriptions for
  technology entities. "EKS" → "Elastic Kubernetes Service", "K8s" →
  "Kubernetes", "RDS" → "Relational Database Service", etc.
- **Weaknesses addressed:** **W1**, surgically. This is the single
  highest-leverage fix for the exact failure mode the MA2 report documents
  (the "Eventual Consistency Cluster" hallucination). Instead of hoping a
  posting corpus contains the right expansion, retrieve the authoritative
  gloss for any acronym the prompt contains.
- **Effort:** Low-to-medium. You curate a subset (a few thousand IT
  entities) rather than ingesting all of Wikidata. Either seed from the
  cloud providers' service catalogs and map to Wikidata for descriptions,
  or use a list like the CNCF Landscape (Tier 1.4) as the seed.
- **Feasibility for MA3:** High. The curated subset is small and stable.
- **Caveats:** Coverage of very new or niche cloud services can be thin.
  Cross-check against the providers' own documentation for entities not
  in Wikidata.

### 1.4 CNCF Cloud Native Landscape

- **Source:** Cloud Native Computing Foundation, GitHub, machine-readable
  YAML/CSV.
- **What it injects:** A curated catalog of ~1,100 cloud-native projects,
  categorised (orchestration, networking, observability, etc.), with
  maturity levels and license info.
- **Weaknesses addressed:** **W1**, specifically for the cloud-native
  domain. When the model needs to mention monitoring tools, it sees the
  actual CNCF-listed options rather than inventing a non-existent APM
  product.
- **Effort:** Low. Small file, well-structured.
- **Feasibility for MA3:** High.
- **Caveats:** Cloud-native only; misses data engineering, ML, frontend,
  enterprise. Inclusion does not imply endorsement (some listed projects
  are early-stage or defunct).

### 1.5 Stack Overflow Annual Developer Survey

- **Source:** survey.stackoverflow.co, annual CSV releases, ODbL.
- **What it injects:** Self-reported technology adoption, co-occurrence
  patterns (which languages/frameworks/cloud tools actually appear
  together), salary by technology and country.
- **Weaknesses addressed:** **W1** (real tech names, correctly spelled),
  **W3** (realistic stack combinations and salary bands by country). The
  co-occurrence signal also helps **W2** by varying stacks.
- **Effort:** Low. Clean CSV, well-documented.
- **Feasibility for MA3:** High.
- **Caveats:** Annual snapshot. Self-reported sample skews toward Stack
  Overflow's demographic (more web, fewer enterprise/legacy). Salary is
  global but heavily US-weighted. **Mentioned by all 9 of 9 LLMs**, which
  is rare and a real signal — but the consensus is partly that everyone
  reaches for it because it is the most visible open dataset, not because
  it is the strongest fit. Use it for stack co-occurrence; treat its
  salary numbers as a sanity band rather than ground truth.

---

## Tier 2 — High value, moderate effort, still fits in MA3

### 2.1 Cloud provider service catalogs (AWS, GCP, Azure)

- **Source:** Provider-published documentation, open for reference (commercial
  redistribution licenses vary; for academic/non-commercial use this is
  fine).
- **What it injects:** Complete, authoritative service names and short
  descriptions for AWS, GCP, Azure.
- **Weaknesses addressed:** **W1**, the direct antidote to the EKS-style
  failure mode. AWS has ~250 services; the entire catalog as a
  retrieval index fits in a few MB.
- **Effort:** Medium. Documentation is massive; you only want the service
  name → canonical short description mapping, not the full docs. Either
  scrape the provider's "all services" landing pages (legal grey area; OK
  for academic) or join the Wikidata subset (Tier 1.3) against the
  providers' published "what is X service" pages.
- **Feasibility for MA3:** Medium-to-high. A focused extraction of the
  service-catalog pages is a one-evening job.
- **Caveats:** New services land monthly; you would need a refresh cadence
  for production. For MA3 a one-shot snapshot is fine.

### 2.2 Sample retrieval from the existing Djinni SFT corpus (self-RAG)

- **Source:** Already on disk (`data/processed/converted.jsonl`, 2,507
  records). Zero new data acquisition.
- **What it injects:** Real recruiter requests paired with the postings
  the teacher-generated, indexed by extracted role + tech-stack tuple, so
  that at inference the top-k most similar past requests can be retrieved
  to provide phrasing variety.
- **Weaknesses addressed:** **W2** primarily (template diversity via
  example-based RAG). ChatGPT's "Tier 6" suggestion in the source file.
- **Effort:** Low. The data is already cleaned and tokenised; you only
  need to embed it and build a similarity index.
- **Feasibility for MA3:** Very high — this is essentially free.
- **Caveats:** Does not address **W1** (model still hallucinates tech
  names because the corpus does too). It is a stylistic helper, not a
  factual one. Pair it with Tiers 1.1-1.4 for the factual side.

### 2.3 Eurostat IT labour market data

- **Source:** ec.europa.eu/eurostat, CC-BY 4.0.
- **What it injects:** EU-level wage statistics by occupation (ISCO-08),
  country, and sometimes region. Closer to the Djinni-corpus geography
  than US-only sources.
- **Weaknesses addressed:** **W3** (salary norms), specifically for
  Europe.
- **Effort:** Medium. Eurostat tables are normalised (good) but require
  joining ISCO occupation codes to recruiter-request role labels (some
  work). DeepSeek's suggestion of linking ISCO-08 codes via job titles is
  correct in principle but adds a normalisation step.
- **Feasibility for MA3:** Medium. The join-ISCO step is the bottleneck.
- **Caveats:** Macro-level (country/region). Not granular to a specific
  tech stack or seniority title. Treat as a band, not a precise figure.

### 2.4 BLS Occupational Employment and Wage Statistics (OEWS)

- **Source:** US Bureau of Labor Statistics, public domain.
- **What it injects:** Annual wage estimates for ~800 occupations at
  national, state, and metropolitan-area level.
- **Weaknesses addressed:** **W3** (salary baselines), US-only.
- **Effort:** Low. Clean public CSV.
- **Feasibility for MA3:** High.
- **Caveats:** US-only and SOC-coded (same broad-occupation problem as
  O*NET). For a Djinni-trained model that mostly sees EU postings, BLS
  is a fallback when Eurostat coverage is thin; not the primary salary
  source.

---

## Tier 3 — Use sparingly; or as one signal in a small ensemble

### 3.1 LinkedIn / Indeed / Glassdoor scraped posting dumps (Kaggle)

- **Source:** Various Kaggle mirrors (arshkon LinkedIn 2023-2024 is the
  most-cited; ~124K postings). 7/9 LLMs recommend something in this
  family.
- **What it injects:** Real recruiter phrasing, structural variety,
  occasional salary and skills fields.
- **Weaknesses addressed:** **W2** (phrasing diversity), partial **W3**
  (the salary and tech-stack fields where they are populated).
- **Effort:** Medium-high. Requires deduplication (job boards repost
  aggressively), HTML stripping, language filtering. The eu-tech-jobs
  lesson applies here: the structured fields (salary, skills, work-type)
  are often sparsely populated. **Verify population rate per field
  before deciding which to index.**
- **Feasibility for MA3:** Medium. The technical work is reasonable, but
  see caveats.
- **Caveats:** **Licensing is the real blocker**, not preprocessing.
  LinkedIn, Indeed, and Glassdoor ToS prohibit scraping; the "open" label
  on Kaggle does not transfer rights to the underlying third-party
  copyrighted text. For an academic deliverable with attribution, this is
  a defensible grey-area use; for anything beyond that, it isn't. Claude's
  position from the source file is reasonable: lower-risk for RAG
  grounding than for training-set inclusion, but a legal-review item before
  shipping. **Also: the model already saw similar postings via Djinni**,
  so the marginal value of *more postings* is modest. Use a small,
  carefully-deduplicated sample (5-10K records) for phrasing variety,
  not a 124K dump in full.

### 3.2 ATS Scrapers Dataset (DeepSeek only)

- **Status:** Mentioned only by DeepSeek, with the claim of "3.2 million
  live jobs from over 86,000 companies, sourced directly from official
  Applicant Tracking Systems". This is a name that reads more like a
  category descriptor than a specific HuggingFace/Kaggle artifact, and
  no other LLM mentions it.
- **Verification needed before ranking:** find the actual dataset, open
  it, check field-population (the eu-tech-jobs lesson). The claim is the
  kind that would have ranked eu-tech-jobs as a top-tier resource on its
  schema alone.
- **Default position:** Treat as unverified until inspected. Do not invest
  effort here until field population is checked.

### 3.3 leadita/tech-stack-datasets (GitHub)

- **Status:** Mentioned by Grok and Manus with claims of 51M-58M
  companies with detected tech stacks. **This is exactly the shape of the
  eu-tech-jobs claim that turned out to be empty in the columns that
  mattered.**
- **Verification needed:** before any retrieval design work, open the
  dataset, sample 200 random rows, and count: what fraction of companies
  have a non-empty primary technology? A non-empty stack-tuple? A
  non-empty industry? If those rates are low, drop it.
- **If the data is dense:** it would address **W3** strongly (company
  tech stacks) and could land in Tier 1.
- **If the data is sparse:** drop. The MA3 timeframe does not have room
  for a second eu-tech-jobs episode.

### 3.4 Common Crawl (5/9 LLMs)

- **Source:** commoncrawl.org, petabyte-scale.
- **Why this is a bad fit for MA3:** Preprocessing cost dominates
  everything else. The MA3 deliverable is two months; "build a job-board
  page detector, scale-process Common Crawl, extract structured postings"
  is a thesis-length project on its own.
- **What it would solve if you had unlimited time:** **W2** (scale and
  freshness for phrasing diversity).
- **Recommendation for MA3:** **Skip.** The fact that half the LLMs
  reach for Common Crawl is a sign they were brainstorming, not
  estimating feasibility. The MA2 lessons-learned discipline (do not
  over-scope) applies here.

### 3.5 Lightcast Skills Taxonomy

- **Status:** Mentioned by DeepSeek, CoPilot, Manus, Z.ai.
- **Caveat:** The Lightcast "Open Skills" project does have a free tier
  (~34K skills) but production-grade API access and the most useful
  features are commercial. ESCO and O*NET together cover most of the
  same ground at zero licensing risk. **Recommendation:** ESCO + O*NET
  first; only reach for Lightcast if a specific MA3 gap appears that
  ESCO+O*NET cannot fill.

### 3.6 StackShare community exports

- **Status:** Mentioned by Grok, CoPilot, Z.ai.
- **Caveats:** Self-reported by companies, often stale (companies change
  stacks faster than they update StackShare), and the community-mirrored
  exports vary in completeness. Z.ai is correct: this overlaps the
  eu-tech-jobs failure mode — the schema promises rich tech-stack
  fields, the population may not deliver. **Verify population rate
  before using.**

### 3.7 Levels.fyi scraped salary data

- **Status:** Mentioned by ChatGPT, CoPilot, Z.ai, Mistral.
- **Caveats:** Same ToS grey area as LinkedIn. Heavily skewed toward
  US-FAANG. Self-reported. Useful for sanity-checking salary bands at
  top-tier companies; not a primary salary source for a Djinni-trained
  EU-leaning model. **Recommendation:** if salary RAG is a priority,
  Eurostat + BLS first; Levels.fyi as a top-end-of-distribution check
  only.

### 3.8 GitHub BigQuery / GH Archive / Libraries.io

- **Status:** Mentioned by ChatGPT, CoPilot, Mistral, Z.ai.
- **Caveats:** Real, large, and clean — but the signal-to-noise ratio
  for "what does a real company tech stack look like" is bad. A
  company's public GitHub repos are a non-representative sample of its
  proprietary stack. Useful for *ecosystem adjacency* (Django →
  PostgreSQL + Celery + Redis) but already covered by the Stack Overflow
  survey co-occurrence data at much lower effort. **Recommendation:
  defer.** Not in the MA3 critical path.

---

## Tier 4 — Skip for MA3

- **Hacker News job posts, AngelList, Remotive, Arbeitnow.** Niche,
  startup-skewed, small. Mentioning a single such corpus does not move
  the needle on any of W1/W2/W3 in a way that ESCO+O*NET+Wikidata does
  not already cover.
- **Synthetic job-description Kaggle datasets** (ravindrasinghrana,
  adityarajsrv). Manus lists these. They are synthetic — they cannot
  ground the model in *real* phrasing, which was the whole point. **Skip.**
- **NIST NICE Cybersecurity Framework.** Useful for a cybersecurity-focused
  product, off-scope for MA3.
- **OpenWebText2 / Dolma / RedPajama.** General web corpora; would need to
  build a job-posting classifier on top, which is back to the Common Crawl
  problem.
- **Country/regional Kaggle salary dumps with strong geographic bias**
  (Naukri / India-only; Malaysia-only). Misaligned with the EU-flavoured
  Djinni corpus. If a regional sweep is part of MA3 evaluation later,
  revisit; for the baseline RAG, skip.
- **"Tech Job Postings Dataset 2026" (CoPilot, Manus).** 1,020 postings,
  academic-use-only license. Too small and too restrictive for the work.
- **Common Crawl** (see 3.4 above). Out of scope on effort grounds.

---

## Recommended MA3 starter corpus

Two retrieval indexes, each small and curated:

**Lookup / normalisation index** (entity-level, exact-match retrieval):

| Source | Role | Effort |
|---|---|---|
| ESCO skills + occupations | Canonical skill names, occupation definitions | Low |
| O*NET Technology Skills + Tools Used files | Canonical software/tool names per occupation | Low |
| Wikidata IT-entity subset (curated) | Acronym → canonical name + one-line description (the EKS fix) | Low-medium |
| CNCF Landscape | Cloud-native tool catalog | Low |
| Cloud provider service catalogs (AWS/GCP/Azure) | Service name + short description | Medium |

Total size: a few tens of MB; embeddable in minutes; the index that does
the **W1** work.

**Similarity / posting index** (chunk-level, nearest-neighbour retrieval):

| Source | Role | Effort |
|---|---|---|
| Existing Djinni SFT corpus (self-RAG) | Phrasing variety on already-seen role/stack tuples | Low (already on disk) |
| 5-10K deduplicated postings from a Kaggle LinkedIn/Indeed dump | Phrasing variety on roles the SFT corpus underrepresents | Medium |

Total size: under 20K chunks; this is the index that does the **W2** work.

**Salary / market index** (optional third index, key-value lookup):

| Source | Role | Effort |
|---|---|---|
| Stack Overflow Developer Survey (latest) | Tech + country + experience → salary band | Low |
| Eurostat IT wage tables | EU country + occupation → wage percentile | Medium |
| BLS OEWS (US fallback) | SOC code + metro → wage percentile | Low |

This index does the **W3** work but is the lowest priority — the salary
question only fires on a fraction of recruiter prompts, and the variance
across geographies is high.

## Datasets to verify before any further commitment

The eu-tech-jobs discipline applies to these — open the data and check
field population before investing more effort:

1. **leadita/tech-stack-datasets** — 51M-58M company claim. Sample, count
   populated tech-stack fields, then decide.
2. **ATS Scrapers Dataset** — DeepSeek-only mention with very large claim.
   Locate the actual dataset (the name is a category descriptor, not a
   specific HF/Kaggle artifact), then sample.
3. **StackShare community exports** — known to be sparse and stale; verify
   coverage before indexing.
4. **Any LinkedIn/Indeed Kaggle mirror used** — verify which structured
   fields (salary, skills, work-type) are actually populated vs. NULL.

## Notes for the report

- The strongest LLM responses in the source file were **ChatGPT, Claude,
  and Z.ai**: all three made the architectural point (split-index
  retrieval, taxonomies as primary lookup), and all three flagged at
  least one feasibility issue (Common Crawl scale, LinkedIn licensing,
  Levels.fyi US-skew). The other six LLMs produced longer lists but
  fewer trade-offs, and at least three (Manus, DeepSeek, Mistral) cited
  datasets that would have ranked highly on schema and need
  field-population verification before they earn a high rank in our
  ranking.
- Stack Overflow Developer Survey being recommended by 9/9 LLMs is a
  positive signal but partly an artifact of visibility. It is the most
  *visible* open dataset in the space, not necessarily the highest
  value. It earns Tier 1 placement here on real merit (clean CSV, tech
  co-occurrence, low effort), not on the unanimous vote.
- O*NET being recommended by 7/9 with ESCO at 4/9 understates how
  complementary the two are. Both is the right answer.
