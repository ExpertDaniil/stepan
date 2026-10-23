# Adaptive execution fixes: verification

Исторический отчёт исходной ветки исправлений. Он не заменяет
[проверку объединённой архитектурной версии](VALIDATION-ARCHITECTURE.md).

The architecture, four research actions, graph policy and validation protocol are retained.
No live API calls or public/private dataset experiments were performed.

- Initial regression run reproduced six assertion failures and one context-budget error.
- Final targeted run: 29 tests passed, including real synthetic fits, tuning from a cached base,
  recovery after invalid code, tool-error repair, exact pipeline reconstruction and publication.
- An additional real-training refit-fallback test passed.
- Release check: 222 tests, 221 passed. One scheduling fixture exposed an unnecessary budget
  dependency in the isolated batch scheduler. It was fixed; its scheduling test and four
  analysis/tuning unit regressions then passed (5 tests). No broad suite failure is hidden.
- git diff --check passed. Requirements and runtime dependencies are unchanged.
- Python 3.12.14; numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, sklearn 1.8.0, joblib 1.5.3.
  This is not the organizers' exact pinned environment; CatBoost is unavailable here.

Final runtime SHA256: `baa1fc4d569b87bb3ecefe7ae85500173e2e47944128c0e9f23ad738f71fe472`.
Flat platform ZIP: 47 files, 198922 bytes; SHA256 `7edd9698406bebca405984bf5bb750b478a0cf366a329366ab70a34cdaed0172`.

For the local quality run, retain the previous budgets and use RESEARCH_MODE=adaptive.
These checks establish execution behavior, not the quality of live Qwen decisions.
