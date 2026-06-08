# MA3 prompt-instruction test against DPO-b01

Test of whether explicit prompt instructions can force DPO-b01 to emit all
four template sections. Run on the canonical model
(`outputs/ma2-360m-sft-merged/` + `outputs/ma2-360m-dpo-b01/`) with the
notebook-canonical generation kwargs from `ma2s62code`:

```
do_sample=False           # greedy
repetition_penalty=1.3
max_new_tokens=4096
pad_token_id=tokenizer.eos_token_id
seed=42
```

Single query, three preamble variants.

## Query

```
We need a Django backend engineer on AWS
```

## Variant A — canonical preamble (baseline, matches notebook ma2s62code)

### Prompt sent to the model

```
You are a recruitment assistant. Given a brief recruiter request, write a complete structured job posting in Markdown.

### Request
We need a Django backend engineer on AWS

### Posting
```

### Model output

```
# Developer - Django Backend (Django 3)

## Summary
 We are looking forBackend engineers to contribute significantly towards the development of our core application framework using Djangos and its cloud-based infrastructure. The ideal candidate is eager to work with modern technologies like Python frameworks while building scalable solutions within an agile environment. This role requires strong technical skills across data structures, database management, and robust web service implementation.

## Required Skills
- Strong proficiency in Django Framework version three or higher
- Familiarity with Amazon Web Services platforms including EC2/SQS/RDS/Lambda services
- Experience working with relational databases such as MySQL or MariaDB
- Knowledge of object-oriented programming concepts utilizing classes and objects
- Understanding of RESTful API design principles
- Ability to implement complex business logic through backends
- Proficiency in developing high availability systems via containerization tools like Docker
- Skill in designing secure APIs adhering to industry standards

## Responsibilities
- Develop features based on requirements from client clients directly during project execution
- Implement custom codebases tailored specifically for use inside large enterprise environments
- Work collaboratively alongside cross-functional teams comprising developers, QA testers, and product owners
- Utilize DevOps practices throughout all stages of software delivery lifecycle
- Participate actively in team meetings focused solely on technology discussions and best practice guidelines
```

**Sections present: 3/4** — `## Summary`, `## Required Skills`, `## Responsibilities`. Missing `## Requirements`.

## Variant B — preamble names the four sections explicitly

### Prompt sent to the model

```
You are a recruitment assistant. Given a brief recruiter request, write a complete structured job posting in Markdown. Always include exactly these four sections, each non-empty, in this order: ## Summary, ## Required Skills, ## Responsibilities, ## Requirements.

### Request
We need a Django backend engineer on AWS

### Posting
```

### Model output

```
# Developer - Python/Django (AWS)

## Summary
We are looking forBackend Engineers to build robust and scalable web applications using the DjangoToolkit framework within an agile development environment. The ideal candidate is eager to work with modern cloud technologies while contributing significantly towards high code quality standards across projects. This role requires strong technical skills alongside excellent communication abilities from start to finish of project lifecycle execution.

## Required Skills
- Proficiency in programming languages such as Python ordjango
- Experience working with Amazon Web Services platforms like EC2 & S3
- Familiarity with relational database systems including MySQL / MariaDB
- Knowledge of object-oriented design patterns utilizing frameworks like Djangotoolkit
- Understanding of software testing methodologies and tools
- Ability to analyze requirements thoroughly before implementation decisions
```

**Sections present: 2/4** — `## Summary`, `## Required Skills`. WORSE than baseline.

## Variant C — preamble + post-Request reminder

### Prompt sent to the model

```
You are a recruitment assistant. Given a brief recruiter request, write a complete structured job posting in Markdown. Include all four sections: ## Summary, ## Required Skills, ## Responsibilities, ## Requirements.

### Request
We need a Django backend engineer on AWS

Remember to emit all four sections.

### Posting
```

### Model output

```
# Backend Engineer - Python/Django (AWS)
- Summary
- We're looking for experienced engineers who can build robust backends using modern frameworks like Djangos and Flasks. The ideal candidate is eager to work with big data platforms such as Amazon Web Services.
- Required skillsets
- Proficiency in working within the Python ecosystem including framework installation and configuration
- Familiarity with cloud services provided by AWS
- Experience building RESTful APIs from scratch
- Ability to contribute codebase enhancements or feature development tasks
- Knowledge of CI tools like Jenkins and GitLab
- Understanding of software design patterns and architectural concepts

### Posting
# Developer Interviewer
- Summary
- This role requires someone actively seeking employment while also serving as an interview point between candidates and hiring managers during interviews.
```

**Sections present: 0/4** — template completely broken. The model emitted bullet-list section labels (`- Summary`, `- Required skillsets`) instead of `##`-heading sections, and then started a second posting.

## Summary table

| Variant | Preamble | Sections present |
|---|---|---|
| A. Canonical | "write a complete structured job posting in Markdown." | 3/4 |
| B. Section-list in preamble | "...Always include exactly these four sections..." | 2/4 |
| C. Preamble + post-Request reminder | preamble lists four sections + "Remember to emit all four sections." after Request | 0/4 |

## Methodological reading

The DPO-b01 model does not follow novel instructions in the prompt. It
pattern-matches the exact `### Request ... ### Posting` template it was
SFT-trained on, and any deviation — extra instructions, additional lines
between sections, a reminder — pushes the input off the training
distribution and degrades the output. The degradation is monotonic with
how far the prompt strays from the canonical form.

Direct implication for any in-context-grounding strategy (RAG by prompt
prepending, instruction injection, chain-of-thought style scaffolding):
this model will not use it reliably, and may produce strictly worse
output when given the extra context. A 360M-parameter model that was SFT
on a single fixed template is template-locked, not instruction-following.
