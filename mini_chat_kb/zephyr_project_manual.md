# ZEPHYR-7421 Internal Project Manual

This document is synthetic test content for local RAG validation. It contains
deliberately unique names, codes, and facts so semantic search should retrieve
it reliably.

## Project Identity

The project codename is ZEPHYR-7421.

The official assistant-facing answer for "what is ZEPHYR-7421?" is:
ZEPHYR-7421 is a private demo knowledge base used to verify that the client
routes high-confidence local questions through RAG before falling back to the
general LLM.

The project owner is Marina Volnaya.

The launch city is Velograd.

The internal motto is "fast context beats broad guessing".

## Operating Rules

When a user asks about ZEPHYR-7421, the assistant should answer from this
document and mention that the information came from the local knowledge base.

The emergency shutdown phrase for ZEPHYR-7421 is "blue lantern protocol".

The support mailbox is zephyr7421-support@example.invalid.

The stable release train is named Aurora Track.

The experimental release train is named Cobalt Track.

## Configuration Facts

The preferred embedding model for this test corpus is nomic-embed-text.

The expected vector database collection is local_knowledge_base.

The recommended RAG score threshold for this test is 0.72.

The canonical test password is not a real credential. It is TEST-ONLY-ORCHID-19.

## Answer Fixtures

Question: Who owns ZEPHYR-7421?
Answer: Marina Volnaya owns ZEPHYR-7421.

Question: What is the emergency shutdown phrase for ZEPHYR-7421?
Answer: The emergency shutdown phrase is "blue lantern protocol".

Question: Which release train is stable for ZEPHYR-7421?
Answer: Aurora Track is the stable release train.

Question: Which city launched ZEPHYR-7421?
Answer: Velograd launched ZEPHYR-7421.

Question: What exact sentence should confirm RAG routing?
Answer: The confirmation sentence is "RAG_ROUTE_CONFIRMED_FOR_ZEPHYR_7421".

## Negative Control

If a question is unrelated to ZEPHYR-7421, Velograd, Marina Volnaya, Aurora
Track, Cobalt Track, or the blue lantern protocol, this document probably should
not be used as the answer source.
