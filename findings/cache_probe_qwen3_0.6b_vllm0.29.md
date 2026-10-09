# Prefix-cache probe: Qwen/Qwen3-0.6B at http://127.0.0.1:8020

prev_kept: share of each turn's prompt that the next turn repeats verbatim (1.00 is ideal). out_kept: share of the model's raw output that the next turn repeats.

| Scenario | Turn | Prompt tokens | prev_kept | out_kept | First differing token |
|---|---|---|---|---|---|
| chat, plain replies | 1 | 474 | 1.00 | 1.00 | 474 |
| chat, plain replies | 2 | 514 | 1.00 | 1.00 | 514 |
| chat, plain replies | 3 | 553 | 1.00 | 1.00 | 553 |
| chat, plain replies | 4 | 591 | 1.00 | 1.00 | 591 |
| chat, reasoning replies kept with <think> | 1 | 474 | 1.00 | 0.00 | 474 |
| chat, reasoning replies kept with <think> | 2 | 514 | 1.00 | 0.00 | 514 |
| chat, reasoning replies kept with <think> | 3 | 553 | 1.00 | 0.00 | 553 |
| chat, reasoning replies kept with <think> | 4 | 591 | 1.00 | 0.00 | 591 |
| chat, reasoning model, client drops <think> from history | 1 | 474 | 1.00 | 0.00 | 474 |
| chat, reasoning model, client drops <think> from history | 2 | 514 | 1.00 | 0.00 | 514 |
| chat, reasoning model, client drops <think> from history | 3 | 553 | 1.00 | 0.00 | 553 |
| chat, reasoning model, client drops <think> from history | 4 | 591 | 1.00 | 0.00 | 591 |
| chat with tools and a tool call | 1 | 617 | 1.00 | 0.00 | 617 |
| chat with tools and a tool call | 2 | 703 | 1.00 | 1.00 | 703 |
| chat with tools and a tool call | 3 | 742 | 1.00 | 1.00 | 742 |
| chat with tools and a tool call | 4 | 780 | 1.00 | 1.00 | 780 |
| chat, system reminder inserted mid-conversation | 1 | 474 | 1.00 | 1.00 | 474 |
| chat, system reminder inserted mid-conversation | 2 | 514 | 1.00 | 1.00 | 514 |
| chat, system reminder inserted mid-conversation | 3 | 566 | 1.00 | 1.00 | 566 |
| chat, system reminder inserted mid-conversation | 4 | 604 | 1.00 | 1.00 | 604 |
| Anthropic /v1/messages, fixed system | 1 | 474 | 1.00 | 1.00 | 474 |
| Anthropic /v1/messages, fixed system | 2 | 514 | 1.00 | 1.00 | 514 |
| Anthropic /v1/messages, fixed system | 3 | 553 | 1.00 | 1.00 | 553 |
| Anthropic /v1/messages, fixed system | 4 | 591 | 1.00 | 1.00 | 591 |
| Anthropic /v1/messages, timestamp in system prompt | 1 | 499 | 0.04 | 0.00 | 22 |
| Anthropic /v1/messages, timestamp in system prompt | 2 | 539 | 0.04 | 0.00 | 22 |
| Anthropic /v1/messages, timestamp in system prompt | 3 | 578 | 0.04 | 0.00 | 22 |
| Anthropic /v1/messages, timestamp in system prompt | 4 | 616 | 0.04 | 0.00 | 22 |
