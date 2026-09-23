# Sample questions (all run against `data/sample_eval_results.csv`)

1. Which model has the highest average accuracy?
2. Which model has the lowest average latency?
3. What is the relationship between cost and accuracy?
4. Which task appears to be the most difficult?
5. Compare the models across accuracy, latency, and cost.
6. Are there any unusual or extreme values in the dataset?
7. What is the average accuracy by model?
8. How many rows per task?

Run one with:

```powershell
& "C:\Users\Aayush\AppData\Local\Programs\Python\Python312\python.exe" run_agent.py data/sample_eval_results.csv "Which task appears to be the most difficult?"
```

Without an `OPENAI_API_KEY` the agent uses its offline dynamic reasoner
(deterministic, no network). Set `OPENAI_API_KEY` (optionally
`OPENAI_BASE_URL` / `OPENAI_MODEL`) to route reasoning + codegen through an
OpenAI-compatible LLM instead.
