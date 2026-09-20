# Lecture 01: Which LLM apps do you use most?

**Final submission deadline: September 20, 2026, 23:59
(Asia/Shanghai, UTC+08:00).** New survey responses will not be accepted after
this deadline.

If you already submitted, check that your Markdown file appears in
[responses/ on the `main` branch](https://github.com/baojian/llm-26-fall/tree/main/surveys/lecture-01/responses).
If your PR is still open, check its review status; do not submit a duplicate.
If you have not submitted, follow the instructions below and open your PR by
the deadline. The teaching team may finish reviewing on-time PRs afterward.

After collection closes and outstanding PRs are resolved, the teaching team
will move this survey to `tasks/l01-tokenization/llm-app-survey/` under
[issue #139](https://github.com/baojian/llm-26-fall/issues/139). The old location
will link to the archive. Accepted responses will be preserved, and no
resubmission is needed. The archive will not accept new survey response files.

Help us understand the class's experience with LLM apps while practicing your
first pull request (PR). Select all apps you regularly use in everyday life
from nine listed apps and Other. There is no limit on the number of selections.
If you do not regularly use any LLM app, leave all boxes unchecked.

The shared [survey issue #6](https://github.com/baojian/llm-26-fall/issues/6)
contains the instructions and a complete response template.

Responses and PRs are public. Use your GitHub username; no real name or student
ID is needed. Share only app names you are comfortable making public.

## Results so far

<!-- survey-results:start -->
Counted from the merged files in [responses/](responses/) on September 20, 2026:
**116 responses**. Multiple selections are allowed.

![Bar chart of the number of students who selected each LLM app](results.svg)

| App | Students |
| :--- | ---: |
| ChatGPT | 102 |
| DeepSeek | 51 |
| Doubao (豆包) | 27 |
| Gemini | 15 |
| Claude | 12 |
| Kimi | 8 |
| Qwen (千问) | 6 |
| Tencent Yuanbao (腾讯元宝) | 2 |
| Zhipu Qingyan (智谱清言) | 2 |
| Codex (Other) | 1 |
| Minimax (Other) | 1 |

Apps written under *Other* get their own bar with the suffix "(Other)".
<!-- survey-results:end -->

## Prepare your response

Copy the [response template](template.md) and mark all applicable choices with
`[x]`. The choices are ChatGPT, Claude, Gemini, DeepSeek, Doubao, Qwen, Kimi,
Tencent Yuanbao, Zhipu Qingyan, and Other. If you select Other, give one app
name. Count the web and mobile versions of the same app as one choice.
One or zero selections is also valid.

Use your GitHub username in lowercase as the filename. For example, the username
`octocat` would submit `surveys/lecture-01/responses/octocat.md`.

## Submit using the GitHub website

1. Sign in to GitHub and open the
   [course repository](https://github.com/baojian/llm-26-fall) on its `main` branch.
2. Select **Add file → Create new file**. GitHub may ask you to fork the
   repository into your account; accept this to create your own copy.
3. Enter `surveys/lecture-01/responses/YOUR_USERNAME.md` as the full filename,
   replacing `YOUR_USERNAME` with your lowercase GitHub username. Paste the
   template into the editor, fill in your answers, and use **Preview** to check
   the formatting. Create your own response file; keep the template unchanged.
4. Choose **Commit changes** or **Propose changes**, with the commit message
   `add lecture 01 survey response`. Use a new branch named `lecture-01-survey`
   if GitHub asks you to choose a branch.
5. Follow **Compare & pull request** or **Create pull request**. Check that the
   destination is `baojian/llm-26-fall`, base branch `main`, and the source is the
   branch containing your response in your fork.
6. Set the PR title to `survey: YOUR_USERNAME` and put `Related to #6` in the
   description. Check that **Files changed** contains only your response file,
   then select **Create pull request**. Keep the PR URL as your submission link.
   The teaching team will review and merge it.

Do not reference the shared issue with `Fixes`, `Closes`, or `Resolves`, since
merging a response should leave the issue open for the other students.

If you need to correct an open PR, edit your response on the same branch in your
fork and commit the correction there; the PR will update automatically. Keep
one PR per student for this survey.

For more help, see GitHub's guides to
[creating a file](https://docs.github.com/en/repositories/working-with-files/managing-files/creating-new-files)
and
[creating a pull request](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request).

## For the teaching team

Apply the deadline above to when the student submitted the response, not when
the teaching team merges it. After September 20, 2026, 23:59 (Asia/Shanghai),
close #6 to mark collection finished and resolve the remaining survey PRs:
review on-time submissions and decline new responses submitted after the
cutoff. Then carry out #139, starting no earlier than September 21, 2026,
00:00 (Asia/Shanghai). Preserve accepted files, totals, and participation
without a second active response directory.

Demonstrate one response and PR during the lecture. Each response has its own
file, so submissions can be merged independently. Review the changed file for
the expected filename, the GitHub username, and an app name if Other is
selected before merging. Any number of selections, including zero, is valid.

After merging a batch of responses, run `uv run python scripts/survey_results.py`
from the repository root. It rewrites the bar chart [results.svg](results.svg)
and the **Results so far** block above, which GitHub shows on this folder's
page and the course website links from the Week 1 row. Include every response,
including those with more than two selections.

The `survey:` title prefix makes these PRs easy to find. Divide review among the
teaching team to handle the class's submissions. After merging, the files in
[responses/](responses/) provide the survey record. Close the shared issue
after collecting the class's responses.
