## 1. Live Talent Supply Calibration (Supply-Side RAG)

Instead of just looking at historical data or abstract skill graphs, retrieve live or near-live data about the *actual candidate pool* to prevent hiring managers from writing "unicorn" job descriptions.

* **Retrieved knowledge:** Internal ATS database size, or API data from talent platforms (e.g., LinkedIn Talent Insights) regarding candidate availability.
* **The RAG Intervention:** If the user asks for "React + Rust + 10 years experience in Lisbon," the RAG system retrieves data showing only 4 people fit this profile.
* **Generated output:** The system automatically flags the bottleneck and generates a calibrated alternative: *"Warning: This combination yields a severely constrained talent pool. Removing 'Rust' expands the local pool by 400%. Here is a revised JD optimizing for trainable adjacent skills..."*

## 2. Codebase & Operations Introspection (Ground-Truth RAG)

Hiring managers often write job descriptions based on what they *think* the team does, not what they *actually* do.

* **Retrieved knowledge:** The company's actual operational surface — recent Jira epics, GitHub repository tags, Confluence architecture docs, or incident reports for the specific team hiring.
* **The RAG Intervention:** The user asks for a "Backend Developer." The RAG system queries the team's recent sprint data and sees a heavy reliance on Kafka and gRPC, plus an upcoming migration to Kubernetes.
* **Generated output:** The system bypasses generic tech skills and injects exact realities: *"In your first 6 months, you will help us migrate our core transaction services to Kubernetes and scale our Kafka event streaming."*

## 3. Proactive Objection Handling (Reputation RAG)

Great job offers sell the company by anticipating candidate hesitations. RAG can analyze your employer brand's weaknesses and automatically insert compensating factors into the offer.

* **Retrieved knowledge:** Recent Glassdoor reviews, internal exit interviews, or rejected offer reasons.
* **The RAG Intervention:** The RAG system identifies that a major candidate drop-off reason for this department is "perceived lack of work-life balance."
* **Generated output:** The JD automatically leads with verified, concrete work-life policies: *"We mandate core collaboration hours from 10 AM to 3 PM, with total flexibility for the rest of your day, and strict no-deploy Fridays to protect your weekend."*

## 4. Platform-Specific Algorithm Optimization (Distribution RAG)

A job offer is useless if nobody sees it. Different job boards parse text differently.

* **Retrieved knowledge:** Current SEO guidelines for Google Jobs schema, LinkedIn Recruiter parsing logic, and Indeed keyword density recommendations.
* **The RAG Intervention:** The system retrieves the structural rules for the platform where the job will be posted.
* **Generated output:** It formats the generated text to guarantee a high match rate with the platform's ATS — ensuring job titles perfectly match the platform's internal taxonomy, even if the "display" title is different.

## 5. Downstream Artifact Generation (Ecosystem RAG)

A job offer is just the first document in a hiring lifecycle. RAG can use the job offer generation as a trigger to simultaneously prepare the rest of the pipeline.

* **Retrieved knowledge:** Internal interview rubrics, technical assessment templates, and 30-60-90 day onboarding plans.
* **The RAG Intervention:** While generating the job description, it maps the required competencies directly to your company's evaluation standards.
* **Generated output:** Alongside the JD, it produces a matched Interview Scorecard and a screening questionnaire. If the JD requires "GraphQL," the scorecard automatically includes the retrieved technical questions for GraphQL.

### The Next-Gen Architecture

The ultimate evaluation target would be a system that balances **Intent vs. Reality**.

| Input/Intent | RAG Source (Reality) | System Action |
| --- | --- | --- |
| **Requirements** | Live Talent API | Calibrates requirements to ensure a viable candidate pool. |
| **Responsibilities** | Jira / GitHub / Slack | Grounds the day-to-day work in actual team data. |
| **Benefits/Pitch** | Exit Interviews / Glassdoor | Injects countermeasures to common candidate objections. |
| **Formatting** | Job Board SEO Rules | Structures the text for maximum algorithmic visibility. |
