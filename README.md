# PySpark and GitHub Actions Practice

This repository is a hands-on learning space for practicing PySpark and GitHub Actions.

## Learning Goals

- Practice data processing and SQL queries with PySpark.
- Learn to create GitHub Actions workflows to automate checks and tests.
- Explore continuous integration (CI): automatically checking code when changes are pushed to GitHub.

## Workflow Output

Run **Program Execution Workflow** from the GitHub Actions tab and enter the Python program path (default: `main.py`).

The job summary displays the captured program output in a code block. If the log is empty or missing, the summary displays a message instead; check the step logs for execution status and failures.

When the artifact upload step succeeds, the full log is also available in the **program-output** artifact as `result.txt` for five days.
