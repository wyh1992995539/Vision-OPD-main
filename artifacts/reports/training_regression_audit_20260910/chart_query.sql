SELECT benchmark, model, COUNT(*) AS n,
       SUM(correct) AS correct, AVG(correct) AS accuracy,
       SUM(finish_reason = 'length') AS length,
       AVG(completion_tokens) AS mean_tokens,
       SUM(inference_error) AS errors, SUM(judge_required) AS judge_required
FROM r4_samples GROUP BY benchmark, model ORDER BY benchmark, model;
