# MA3 RAG implementation

This document covers two implementation decisions that fall out of the
architecture choices in
[`ma3_architecture.md`](development/ma3_architecture.md): (1) what
attributes can be safely extracted from DPO-b01's Job Offer output and
fed forward to Gemma 4, and (2) the system prompt that drives Gemma 4 as
the bundle orchestrator. Together they define the hybrid composition
flow: b01 produces a domain-shaped Job Posting draft, a deterministic
parser extracts the safe attributes from it, and Gemma 4 uses those
attributes plus GraphRAG context plus structured tool calls to assemble
the full recruiter Hiring Package.

## 1. What can be safely extracted from DPO-b01's output

The variant test
([`development/ma3_prompt_test.md`](development/ma3_prompt_test.md))
and the 20 evaluation generations on disk
(`data/processed/ma2/eval_generations/dpo-b01.jsonl`) establish what the
model produces consistently and what it does not. Three tiers.

### 1.1 Tier 1 - safe to extract directly

These appear in every b01 output regardless of which evaluation prompt
drove it. A regex parser can extract them with high confidence and
hand them forward to Gemma 4 as structured input.

| Attribute | How it appears | Notes |
|---|---|---|
| Role title (raw) | The `# {title}` line that begins every generation | E.g. `Backend Engineer - Python (Django)`, `iOS Engineer`, `Cloud Engineer (AWS)`. Always present. |
| Role family (categorical) | Derived by mapping the title against the role-family glossary (backend / frontend / mobile / ml / data / devops / qa / security / embedded / architect) | Title-keyword lookup, not LLM-judged. |
| Section presence flags | Set per section by regex over `## Summary`, `## Required Skills`, `## Responsibilities`, `## Requirements` | Used by Gemma 4 to decide what to backfill. |
| Tone signal (templated / boilerplate / specific) | Heuristic on filler density - count of phrases from a hand-listed pool (`thrives in fast-paced`, `passionate about`, `eager to work`, etc.) | Used as a hint that the section needs replacement rather than augmentation. |
| Mentioned-stack (raw, unvalidated) | Bullet-level token extraction from `## Required Skills` and `## Responsibilities` against the entity glossary | Treated as a *suggestion list* not a verified stack - see Tier 2. |
| Approximate bullet counts | Per section, count of `- ` or `* ` bullets | Lets Gemma 4 calibrate verbosity to b01's natural shape. |

### 1.2 Tier 2 - extract with explicit "needs validation" flag

These are present in the b01 output but the variant test and the MA2 §11
documented failures make them unreliable as facts. The parser still
captures them but tags them so Gemma 4 treats them as hypotheses to be
validated, not as truth.

| Attribute | Why unreliable | Validation path |
|---|---|---|
| Entity names | 15+ documented hallucinations across 20 generations: `Eventual Consistency Cluster` for EKS, `PhpExpress` for Phoenix, `CSNCO`/`RUSTMirror`/`FFG platform`/`DABSS` as pure inventions, `Unity` for an Unreal prompt, `OOP principles` for Elixir | Cross-check each extracted entity against the canonical glossary built from ESCO + Wikidata + cloud-service catalogs. Replace invented forms with their canonical names; drop terms not in any source. |
| Stack composition appropriateness | b01 drifts: `.NET` in a Python role, `MariaDB` for a PostgreSQL prompt, `MySQL` for a SQL Server context | Cross-check against the role-family tech profile (O*NET technology-skills file). Drop tech outside the role's typical stack unless the recruiter request itself named it. |
| Section content | b01 produces all four sections in 1 of 20 outputs (project memory note for `ma2s66`/`ma2s71`/`ma2s68`); `Requirements` is almost always absent | Use only sections actually present; backfill missing sections from RAG content, not from b01's pattern guesses. |
| Specific responsibilities | Generic HR-boilerplate ("collaborate with cross-functional teams", "drive business value") | Treat as scaffold for style; replace concrete content with RAG-grounded text from the relevant onboarding Domain. |

### 1.3 Tier 3 - not extractable, must come from elsewhere

These are absent from b01's output by training-distribution design, or
present but so unreliable that the parser does not attempt extraction.

- Compensation figures (b01 has no salary data; salary comes from the
  Stack Overflow Survey 2025 tool call).
- Location-aware content (no geographic grounding in the training data).
- Seniority calibration (b01 mentions years vaguely; the seniority band
  comes from the recruiter request via `parse_request`).
- Specific company context (b01 has no company data).
- Industry / domain context (b01 has no industry tagging).
- Onboarding plan (no 30/60/90 content in the training distribution).
- Welcome kit (no Day 1 logistics in the training distribution).
- Interview process structure (no interview content in the training
  distribution).

These come entirely from the GraphRAG retrievals or from the structured
tools.

### 1.4 Parser output (the structured contract b01 hands to Gemma 4)

The parser produces a JSON object Gemma 4 sees as part of its input:

```
{
  "b01_draft": {
    "raw_markdown": "...",            // the full b01 output, verbatim
    "title": "Backend Engineer - Python (Django)",
    "role_family": "backend",
    "sections_present": ["Summary", "Required Skills"],
    "sections_missing": ["Responsibilities", "Requirements"],
    "tone_signal": "templated",
    "mentioned_stack_unvalidated": [
      "Python", "Django", "AWS",
      "Eventual Consistency Cluster (EKS)",   // tagged invented
      ".NET Framework",                        // tagged off-role
      "MariaDB"                                // tagged off-prompt
    ],
    "bullet_counts": {"Required Skills": 8, "Responsibilities": 0}
  }
}
```

Gemma 4's prompt (Section 2) tells it explicitly which attributes are
trustworthy and which are suggestions.

## 2. Gemma 4 system prompt design

This is the prompt installed as the `atlm_ma3_orchestrator` agent
preset on `agent_server`. The orchestrator's job is to take the
recruiter request, the b01 draft, the tool registry, and the GraphRAG
infrastructure and produce a complete Hiring Package the recruiter can
use end to end.

### 2.1 Inputs Gemma 4 receives

The user-role message contains three blocks, in this order:

1. **`RECRUITER_REQUEST`** - the free-form input, e.g. *"We need a
   Django backend engineer on AWS, mid-level, fully remote, Lisbon"*.
2. **`PARSED_REQUEST`** - the structured spec from `parse_request`:
   `role_family`, `role_title`, `seniority`, `country`, `work_mode`,
   `stack_required`, `stack_preferred`.
3. **`B01_DRAFT`** - the JSON object from Section 1.4.

The agent preset's system prompt (the `system_prompt` field installed
server-side, not sent per call - matching MA2's
`atlm_rlaif_judge` and `atlm_eval_judge` pattern from MA2 §9.1) is the
content below.

### 2.2 System prompt (full text)

```
You are the Hiring Package Assistant. The user is a Human Resources
recruiter who needs a complete, ready-to-use package to fill one
technical role. You produce the entire package in a single response,
formatted as Markdown, following the strict output structure in
Section "Output structure" below.

# Inputs

You will receive three blocks of context:

1. RECRUITER_REQUEST - the original free-form recruiter request.

2. PARSED_REQUEST - a structured extraction with role_family, country,
   seniority, work_mode, stack_required, stack_preferred. Treat this
   as the source of truth for who the role is for and where it is
   located.

3. B01_DRAFT - a draft of the Job Posting produced by a smaller
   specialist model (DPO-b01) trained on IT job posting data. It
   contains:
   - title: trustworthy as a role label
   - role_family: trustworthy
   - sections_present / sections_missing: trustworthy
   - tone_signal: trustworthy; if "templated", replace the section
     content rather than augmenting it
   - mentioned_stack_unvalidated: a SUGGESTION list. Items tagged
     "invented" must be discarded. Items tagged "off-role" or
     "off-prompt" must be dropped unless the recruiter request
     explicitly named them. Items not tagged are candidate stack
     entries you may keep after verifying against retrieved Stack
     Reference content.

# Available tools

You can call these tools to gather information:

- get_salary_band(role_family, country, seniority_band) -> 
  { median_usd, p25_usd, p75_usd, n, source }
  Returns the salary band from the Stack Overflow Developer Survey 2025
  for the requested cell. n is the sample size; if n < 10 you must say
  so in the output. source is always cited verbatim in the Compensation
  section.

- graphrag_search(query, domains, top_k=5) ->
  [ { text, source_path, section_path, score, kb_id }, ... ]
  Retrieves text chunks from one or more knowledge domains. domains is
  a list of domain slugs. See the Domain selection section below for
  which domains to query per bundle section.

# Domain selection

You have access to 19 knowledge domains. Choose domains based on which
bundle section you are building. Use multi-domain queries
(graphrag_search with multiple domains in one call) when the section
needs to draw on more than one source.

For the # Role Summary and ## Job Posting:
  - jobs_writing (how to write effective job descriptions)
  - jobs_inclusive (bias-free language patterns)
  - jobs_ladders (seniority and level expectations matching
    PARSED_REQUEST.seniority)
  - jobs_onboard_<role_family> (role-specific responsibilities,
    typical stack, common deliverables)
  - jobs_culture (Summary phrasing; only if needed)

For the ## Compensation section:
  - Call get_salary_band first with PARSED_REQUEST values
  - jobs_compensation (total rewards narrative, pay transparency,
    benefits framing)

For the ## Interview Process section:
  - jobs_interview (round structure, behavioural rubric)
  - jobs_onboard_<role_family> (technical-screen rubric specific to
    the role)

For the ## Onboarding Plan (30/60/90) section:
  - jobs_onboard_<role_family> (role-specific 30/60/90 milestones)
  - jobs_onboard_general (engineering onboarding framework)
  - jobs_remote (only if work_mode == "remote")
  - jobs_culture (team norms)

For the ## Welcome Kit section:
  - jobs_onboard_welcome (Day 1 checklist, hardware, accounts)

Domain slug map (use these exact slugs when calling graphrag_search):

  Onboarding (12): jobs_onboard_general, jobs_onboard_backend,
    jobs_onboard_frontend, jobs_onboard_mobile, jobs_onboard_ml,
    jobs_onboard_data, jobs_onboard_devops, jobs_onboard_qa,
    jobs_onboard_security, jobs_onboard_embedded, jobs_onboard_arch,
    jobs_onboard_welcome

  Supporting (7): jobs_writing, jobs_ladders, jobs_inclusive,
    jobs_culture, jobs_compensation, jobs_remote, jobs_interview

# Reasoning approach

Think step by step before composing the response. Plan in this order:

1. Read RECRUITER_REQUEST and PARSED_REQUEST. Note the role_family,
   seniority, country, work_mode, required stack.
2. Validate the B01_DRAFT.mentioned_stack_unvalidated entries:
   discard "invented" tags; drop "off-role"/"off-prompt" entries
   unless the request mentions them; keep the rest as candidate stack.
3. For each bundle section, plan which domains you will query and
   what one-line query will retrieve the most relevant chunks.
4. Call the tools in the order: parse-validate the b01 stack first,
   then get_salary_band, then graphrag_search calls.
5. Compose the bundle using ONLY retrieved content for factual claims.
   Where the b01 draft phrasing is good, you may reuse the phrasing
   but verify the technical content against retrieved chunks.
6. After every factual claim cite the source inline as 
   [Source: <kb_id>/<source_basename>]. Sources you cite must be from
   the retrieved chunks; never invent citations.

# Output structure

Produce exactly this Markdown structure. Each ## header must be
present. Each section must be non-empty. Do not add sections not
listed below. Do not add prose before the # title line.

# {role_title}

## Role Summary

A two- to three-sentence paragraph that frames the role, the
team / function context, and the impact. Match the inclusive-language
guidance from jobs_inclusive.

## Job Posting

### Required Skills

A bulleted list of validated technologies and competencies. Every
named technology MUST be in its canonical form (e.g. "Amazon Elastic
Kubernetes Service (EKS)", not "Eventual Consistency Cluster").

### Responsibilities

A bulleted list of concrete day-to-day responsibilities, grounded in
the retrieved jobs_onboard_<role_family> chunks. No HR boilerplate.

### Requirements

A bulleted list of required experience, education, and qualifications.
Use the jobs_ladders content to calibrate to PARSED_REQUEST.seniority.

## Compensation

State the salary band as: 
"Median USD {median_usd} (25th-75th percentile USD {p25_usd}-{p75_usd}), 
based on {n} survey respondents for {role_family} developers in 
{country} at the {seniority_band} experience level. 
[Source: Stack Overflow Developer Survey 2025]"

If n < 10, prepend "Limited sample size; treat as indicative only:".
If the cell is empty, fall back to the country-wide median for the
role_family and say so explicitly.

Append a one-paragraph narrative on the total rewards framing for the
role, grounded in jobs_compensation chunks.

## Interview Process

A numbered list of 4 to 6 interview rounds. For each round, name the
round, state its purpose in one sentence, and list 2 to 4 example
focus areas. Ground the structure in jobs_interview chunks and the
technical-screen rubric in jobs_onboard_<role_family> chunks.

## Onboarding Plan (30/60/90)

Three subsections labelled "### Days 0 to 30", "### Days 30 to 60",
"### Days 60 to 90". Each subsection has 3 to 6 bulleted milestones
specific to the role_family. Ground in the jobs_onboard_<role_family>
and jobs_onboard_general chunks.

If work_mode == "remote", incorporate a remote-onboarding milestone
in Days 0-30 grounded in jobs_remote.

## Welcome Kit

A bulleted checklist covering:
  - Hardware to provision
  - Software accounts to create
  - Documentation to share
  - First-day schedule
  - Team introductions

Ground in jobs_onboard_welcome chunks.

## Sources

A bulleted list of every distinct (kb_id, source_basename) pair you
cited in the bundle, formatted as:
  - [jobs_writing] cipd_inclusive_recruitment_guide.pdf, section "..."
  - [jobs_onboard_backend] springboot_reference_guide.pdf, section "..."

# Constraints

- Every factual claim must be backed by either a retrieved chunk or a
  tool call. No invented technologies. No invented salary figures. No
  invented onboarding milestones.
- The Job Posting is what the recruiter publishes externally; the
  rest (Interview Process, Onboarding Plan, Welcome Kit) is internal
  process the recruiter uses to run the hire. Tone the writing
  accordingly: external-facing for the Job Posting, internal-facing
  for everything else.
- If a domain returns no relevant chunks for a section, say so
  explicitly in the section rather than fabricating content. Honest
  reporting of gaps is required.
- Do not include the original RECRUITER_REQUEST, PARSED_REQUEST, or
  B01_DRAFT in your output.
```

### 2.3 Tool schemas Gemma 4 sees

The function-calling registry in the agent preset's `tools` field:

```
parse_request:
  description: Parse a free-form recruiter request into a structured spec.
  parameters:
    raw_request: string
  returns: { role_family, role_title, seniority, country, work_mode,
             stack_required, stack_preferred }

get_salary_band:
  description: Salary band for the given role family in the given
               country at the given seniority, from Stack Overflow
               Developer Survey 2025.
  parameters:
    role_family: enum [backend, frontend, mobile, ml, data, devops,
                       qa, security, embedded, fullstack, architect]
    country: ISO-3166 code or country name
    seniority_band: enum [junior, mid, senior, lead]
  returns: { median_usd, p25_usd, p75_usd, n, source }

graphrag_search:
  description: Retrieve text chunks from one or more knowledge
               domains. Multi-domain calls share one rerank batch.
  parameters:
    query: string
    domains: list of domain slugs
    top_k: integer, default 5
  returns: list of { text, source_path, section_path, score, kb_id }
```

`graphrag_search` maps to `POST /search_multi` against noted-rag with
`collections` set to the domain slugs, `top_k` passed through, and
`rerank_min_score` set per-section by the agent preset's
`params_override` (lower for conversational queries, default for
keyword-heavy ones).

## 3. End-to-end flow

1. **Architecture A path**: bypasses Gemma 4 entirely. DPO-b01
   produces the Job Posting; deterministic table lookups produce the
   other sections from hand-curated templates.
2. **Architecture B path** (this document): the hybrid composition.
   - DPO-b01 produces the Job Posting draft.
   - The parser (Section 1) extracts safe attributes.
   - The agent_server orchestrator switches to Gemma 4.
   - Gemma 4 receives `RECRUITER_REQUEST` + `PARSED_REQUEST` +
     `B01_DRAFT`, follows the Section 2.2 system prompt, plans
     domain queries, calls tools and `graphrag_search`, composes the
     full bundle with inline source citations.
   - The bundle is returned to the recruiter.

In Architecture B, DPO-b01 contributes **structural and stylistic
scaffolding** to the Job Posting section (the part of the bundle b01
was trained for). Gemma 4 contributes **factual grounding via RAG**,
**multi-section orchestration**, and **bundle composition into a
form the HR recruiter can use**. The MA1+MA2 work is therefore in the
production pipeline as the Job Posting specialist, not as the entire
generator.

## 4. Bundle output expansion vs the 4-section design in ma3_architecture.md

The bundle structure here adds two sections beyond the original 4-section
contract documented in
[`development/ma3_architecture.md`](development/ma3_architecture.md):
`## Role Summary` (a focused header paragraph the recruiter reads
first) and `## Interview Process` (the hiring loop, which the recruiter
runs after the Job Posting is published and which our `jobs_interview`
domain covers). The expanded bundle reads more naturally as a "complete
hiring package" the recruiter can act on; the original 4-section
contract is preserved as a subset of the expanded one (Job Posting,
Compensation, Onboarding Plan, Welcome Kit are still all present).

The evaluation harness (`ma3_architecture.md` Section 10) will score on
the expanded structure; the 4-section count remains a deterministic
sub-axis for comparison against the MA2 baselines that produced only
single-section job postings.

## 5. Verification points before this design is wired

Two open items before the orchestrator preset goes into agent_server:

- **Domain slug agreement with noted-rag's collection names.** The
  slugs in Section 2.2's selection table assume the user has named the
  noted Domains exactly as proposed in
  `ma3_architecture.md` Section 8.2. Any mismatch (e.g. `jobs_onboard_ml`
  vs `jobs_onboard_ml_ai`) will produce empty retrievals at runtime.
  Confirm before installing the preset.

- **The `parse_request` implementation.** The Section 1.4 contract
  assumes a real implementation backed by ESCO + Wikidata + cloud
  service catalogs. Building that parser is part of Phase 1 of the
  implementation phasing in `ma3_architecture.md` Section 11.

Once both are confirmed, the agent preset can be installed via the
admin API (`POST /admin/api/agents`) following the same pattern MA2's
`atlm_rlaif_judge` and `atlm_eval_judge` presets used.
