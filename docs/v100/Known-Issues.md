# Known issues

Problems seen while running these models, what causes them, and what to do about them.
Each entry says whether the cause is in this engine or in the model.

## SenseNova-Loop-9B: a stray Chinese token can start an endless repeat

**Cause: the model, not the engine.** The model's own reference implementation (its
`modeling_sensenova_loop.py`, run under transformers in fp32) does the same thing on the same
text. Nothing in this engine needs a fix.

### What happens

In a long English reasoning trace, the model writes one Chinese token and then repeats it until
`max_tokens` runs out.

This has been seen once, on HLE question `672bca9fab085c84b90e223d` (Biochemistry, multiple
choice). The run had thinking on and used the model's shipped sampling settings: temperature 1.0,
top_p 0.95, no top_k, from its `generation_config.json`. After about 4,400 tokens of normal
English reasoning, mid-sentence ("…For example, the classic"), the model wrote 杆 and
repeated it about 500 times. Then it switched to ∀ ("for all") and repeated that about 22,700 times, until the
32,768-token limit.

### Why

**The first token is a language switch.**
- 杆 is Chinese for "rod" (pole, staff, shaft). The model's reasoning at that point was about
  structures that are usually described as rod-shaped.
- At that position the model's distribution holds Chinese translations of its English options:
  例子 "example" and 案例 "case". In total, Chinese tokens have 1.8% of the probability there.
- 杆 itself has probability 1e-4 (rank 302). That is inside the 653 tokens that top_p 0.95 allows,
  so the shipped settings can draw it about once in 10,000 tries.
- Nothing in the input is Chinese: the prompt, the system prompt, the chat template and the 4,400
  tokens before are all English.

**The repeat comes from the model.**
- Once one 杆 is in the text, the model gives the next 杆 a probability of 0.91, then 0.61, 0.82
  and 0.86.
- Starting from the same text plus one 杆, 25 of 50 sampled continuations got stuck on it. This
  engine and the reference give the same count.
- The repeat comes from the second pass over the looped layers. With one pass over layers 4-27
  instead of the trained two, the repeat probability is 0.16 and none of 50 continuations get
  stuck. That run only locates the cause. It is not a fix: the model was trained with two passes.
- It needs both this token and this place:
  - 30 other rare tokens at the same position: 0 of 600 continuations stuck;
  - 杆 at 50 other positions: 0 of 1,000;
  - one rare token (rank 20 to 2,000) put into 274 other position and rank-band pairs in other
    texts: 0 of 7,700.
- Qwen3.8-Flash-Next uses the same tokenizer. On the same text it gives 杆 a probability of 4.4e-7
  and repeats it with 0.0006, and none of 50 continuations get stuck.

**Why the engine is not the cause.**
- The log-probability of 杆 at that position is −9.2150 in this engine and −9.2153 in the fp32
  reference.
- The sampler draws with the reference's frequencies: 82,008 draws at three positions, G-test
  p 0.49-0.65.
- Over 205,860 generated tokens, decode matches the fp32 reference: the 99th-percentile log-prob
  difference is 0.0125 nats, and it does not grow with position out to 32k tokens.
- It also matches with the original run's server settings and long prefills running alongside:
  75,015 tokens, largest difference 0.074 nats.
- The model's configuration is not the cause. The released YaRN RoPE scaling and plain RoPE, and
  fp32, bf16 and fp16, all give the same collapse probabilities: P(杆) 0.99e-4 to 1.09e-4, first
  repeat 0.87-0.91.
- Nothing hides, filters or biases tokens:
  - no watermark, logit bias or grammar in that run;
  - no suppressed or banned tokens in the model's generation config;
  - 杆 and ∀ are normally trained tokens (compared with the checkpoint's 243 never-trained
    embedding rows).

### How often

- Collapses: 1 in 510,340 generated tokens of HLE reasoning at the shipped settings. That is 0.002
  per 1,000 tokens, with a 95% confidence interval of 0.00005 to 0.011. A single event gives only a
  rough rate.
- Language switches: 3 Chinese tokens in about 519,000 tokens of English output, all on this one
  question. The other two were followed by English straight away.

### What to do

- **Set `top_k` in the request.** With `top_k` 50 or 20, the first 杆 (rank 302) cannot be drawn
  at that position.
  - OpenAI `/v1/chat/completions`: add `"top_k": 20` to the request body. The OpenAI Python client
    sends it through `extra_body`.
  - Anthropic `/v1/messages`: `top_k` is a standard field.
- **What the model's authors suggest.** Their inference code defaults to temperature 0.7 and
  top_p 0.9, and transformers then applies top_k 50. Their documentation suggests
  `--top-k 20 --repetition-penalty 1.05`.
- **Current server default.** `scripts/serve_sensenova_loop_v100.sh` sets top_k 20 for requests
  that set none (`SENSENOVA_TOP_K`, -1 turns it off). A request's own `top_k` wins.
- **Limits of this workaround.**
  - Its effect on answer accuracy has not been measured.
  - It stops the first stray pick. It does not break a repeat that has already started: under the
    authors' settings, the next four 杆 are almost certain (P 1.0 to two decimals) once one is in
    the text.
  - A repetition penalty might break such a repeat. That has not been measured either.
- **If a reply gets stuck,** it ends at `max_tokens`. A long run of one token is easy to detect
  on the client side; a retry draws a new sample.
