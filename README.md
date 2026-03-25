# Project Retriva's documentation

## Introduction

This project is an attempt to kill two birds with one stone:

1. To challenge myself with a non-trivial project that would allow me to put into practice what I have learned in the field of AI over the past few years through self-study and the courses I have attended. Indeed, if there is one thing I have no doubt about, it is that in disciplines such as electronic and computer engineering, it is essential to get one’s hands dirty in order to truly internalize concepts that have only been studied on paper.
2. To find a solution for querying, in natural language, the knowledge base (KB) built on the document archive of my company’s R&D department, developed over the decades and implemented as a website based on [MediaWiki](https://www.mediawiki.org/wiki/MediaWiki).

The need described in point 2 quickly led me toward developing a conversational agent equipped with RAG capabilities, which I named Retriva. Retriva is supposed to be a system perfectly aligned with the technologies referred to in point 1. In addition, throughout the project I have tried to use AI tools to help me tackle its various phases, as will be described in greater detail in the rest of the document.

In the section dedicated to the design, the system requirements and specifications are discussed in greater detail. For the time being, I will simply recall that, from the very beginning, the project was conceived to satisfy a strict confidentiality requirement: use of external services was ruled out in order to ensure that the information contained in the knowledge base would never leave the company’s perimeter. All things considered, the system could therefore — with the appropriate adaptations — be reused in many enterprise scenarios. The need to consult knowledge accumulated over time while offering a UX that keeps pace with the times is, in fact, very widespread among companies, regardless of the sector in which they operate.

Before launching the project, I had tried to meet my needs by adopting a “minimum effort, maximum result” approach. In practice, I attempted to solve the problem using tools such as Cheshire Cat or AnythingLLM. While these tools are more or less ready to use, they obviously do not allow one to learn much and, above all, they offer only limited control over the underlying mechanisms, first and foremost the process of building the proprietary knowledge base. This therefore led me to decide to take the bull by the horns and try to create something truly satisfactory for my needs.

## Preliminary roadmap

Generally speaking, the project's development plan includes three major phases:

1. In a development environment, implementation of a proof-of-concept (PoC) that uses a public Mediawiki site and whose goal is to validate the overall architecture.
2. Creation a usable but unoptimized first version.
3. Gradual integration of more advanced technologies for improving retrieval.
4. Move to the "real" knowledge base, i.e. the one generated from the confidential Mediawiki website and deploy Retriva in the business environment.
5. Add advanced functionality such as agent capabilities and user role management.

## Design

### AI-assisted design sessions

* [#1](./design_sessions/Retriva-design_1.pdf)
