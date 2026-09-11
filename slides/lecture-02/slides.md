<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">CS40008.01</p>

# N-gram Language Models

<p class="subtitle">Lecture 02 – NLP and LLMs (CS40008.01)</p>

<p class="byline">Baojian Zhou<br>School of Data Science<br>Fudan University<br>September 16, 2026</p>

Note:
Ported from Fudan Spring Lecture 02, https://baojian.github.io/llm-26/slides/lecture-02-slides/. Open with the question: how can a model assign a probability to a sentence? Three 45-minute periods: probabilistic N-gram LMs and evaluation in periods 1–2, smoothing and neural probabilistic LMs in period 3.

---

<!-- .slide: class="outline-slide" id="outline-ngram" -->

## Outline

<ul class="outline-topics">
<li aria-current="step">Probabilistic N-gram LMs</li>
<li>Evaluating LMs and Perplexity</li>
<li>Smoothing N-gram LMs</li>
<li>Neural Probabilistic LMs</li>
</ul>

Note:
Period 1: the first topic. Return to this outline at each transition. Source: Spring Lecture 02 slide 2, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/1.

---

<!-- .slide: id="assign-probabilities" -->

## Assign probabilities to sentences

<p><strong>Speech recognition</strong></p>
<p>$P($<span class="text-good">It's hard to recognize speech</span>$) \gt P($<span class="text-bad">It's hard to wreck a nice beach</span>$)$</p>

<p><strong>Machine translation (MT):</strong> “他向记者介绍了主要内容” is translated into 4 candidates.</p>
<ul>
<li>$S_1$ = <span class="text-good">He briefed reporters on the main contents of the statement</span></li>
<li>$S_2$ = He introduced reporters to the main contents of the statement</li>
<li>$S_3$ = He briefed to reporters the main contents of the statement</li>
<li>$S_4$ = <span class="text-bad">He to reporters introduced main content</span></li>
</ul>

Note:
Each task needs a score that prefers fluent sentences. In speech recognition the two transcriptions sound alike; only a language model separates them. The MT candidates are all translations of the same Chinese sentence. Source: Spring Lecture 02 slide 3, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/2.

---

<!-- .slide: id="assign-probabilities-2" -->

## Assign probabilities to sentences

<p>A good MT model should have</p>
<p>$P($<span class="text-good">$S_1$</span>$) \gt P(S_2) \approx P(S_3) \gt P($<span class="text-bad">$S_4$</span>$)$</p>

<p><strong>Spell correction</strong></p>
<ul>
<li>The office is about fifteen <span class="text-bad">minuets</span> from my house</li>
<li>$P($<span class="text-good">about fifteen minutes from</span>$) \gt P($<span class="text-bad">about fifteen minuets from</span>$)$</li>
</ul>

Note:
“Minuets” is a real word, so a dictionary alone does not catch it. A language model prefers the phrase with “minutes.” Source: Spring Lecture 02 slide 3, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/2.

---

<!-- .slide: id="unknown-distribution" -->

## LLMs approximate an unknown distribution

<p><strong>Real text-based data</strong> (languages, code, etc.) induces a distribution over token sequences:</p>
<p>$$\mathbf{w}_{1:n} = (w_1,\dots,w_n), \quad \mathbf{w}_{1:n} \sim p_{\text{data}}(\cdot),$$</p>
<p>where <span class="text-bad">$p_{\text{data}}$ is unknown</span>. We only observe a corpus (samples):</p>
<p>$$\mathcal{D}=\{\mathbf{w}^{(i)}\}_{i=1}^N,\quad \mathbf{w}^{(i)}\overset{\text{i.i.d.}}{\sim} p_{\text{data}}(\cdot).$$</p>

Note:
The population of all text is never observed; a corpus is a sample from it. Keep the i.i.d. assumption explicit; it is an idealization. Source: Spring Lecture 02 slide 4, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/3.

---

<!-- .slide: id="model-distribution" -->

## LLMs approximate an unknown distribution

<p>We build and train a model distribution (LLMs) $p_\theta(\mathbf{w})$ such that</p>
<p>$$p_\theta(\mathbf{w}) \approx p_{\text{data}}(\mathbf{w}).$$</p>

<blockquote><p><strong>In $p_\theta$,</strong> sequences that look like real language should get <span class="text-good">higher</span> probability than corrupted or random ones:<br>$p_\theta(\text{“natural sentence”})\;\gt\;p_\theta(\text{“random sentence”}).$</p></blockquote>

Note:
This is the whole modeling goal in one line. Everything that follows is about how to parameterize $p_\theta$ and how to fit it. Source: Spring Lecture 02 slide 4, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/3.

---

<!-- .slide: id="training-objective" -->

## Training objective: make $p_\theta \approx p_{\text{data}}$

<p>Ideal goal (distribution matching) is to make $p_\theta \approx p_{\text{data}}$. One principled way is to find a $\theta$ that minimizes the KL divergence:</p>
<p>$$\theta^\star \in \arg\min_\theta \mathrm{KL}\big(p_{\text{data}} \,\|\, p_\theta\big).$$</p>
<p>This is equivalent to maximizing expected log-likelihood:</p>
<p>$$\arg\min_\theta \mathrm{KL}(p_{\text{data}}\|p_\theta) \;\Longleftrightarrow\; \arg\max_\theta \mathbb{E}_{\mathbf{w} \sim p_{\text{data}}}\big[\log p_\theta(\mathbf{w})\big].$$</p>

Note:
Expand the KL divergence: the entropy of $p_{\text{data}}$ does not depend on $\theta$, so minimizing KL equals maximizing the expected log-likelihood under the data distribution. Source: Spring Lecture 02 slide 5, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/4.

---

<!-- .slide: id="training-objective-2" -->

## Training objective: make $p_\theta \approx p_{\text{data}}$

<p>Since $p_{\text{data}}$ is unknown, use the dataset $\{\mathbf{w}^{(i)}\}_{i=1}^N$ to approximate the expectation:</p>
<p>$$\max_\theta \frac{1}{N}\sum_{i=1}^N \log p_\theta (\mathbf{w}^{(i)}).$$</p>
<p>Let $\mathbf{w}^{(i)} := (w_1^{(i)},\ldots,w_{n_i}^{(i)})$. The LM factorization (what the model actually learns):</p>
<p>$$p_\theta(\mathbf{w}^{(i)})=\prod_{t=1}^{n_i} p_\theta \left(w_t^{(i)} \mid w_{1:t-1}^{(i)} \right).$$</p>
<p class="caption">So training teaches the model to predict the next token like real text → sampling from $p_\theta$ produces human-like sequences.</p>

Note:
The empirical average replaces the expectation. The factorization is exact by the chain rule; the modeling choice is how each conditional is represented. Source: Spring Lecture 02 slide 6, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/5.

---

<!-- .slide: id="training-samples" -->

## Training samples from real-world

<p>$$\mathcal{D}=\left\{\mathbf{w}^{(i)}\right\}_{i=1}^{N}=$$</p>
<ul>
<li>$\mathbf{w}^{(1)}$ : It's hard to recognize speech.</li>
<li>$\mathbf{w}^{(2)}$ : He briefed reporters on the main contents of the statement.</li>
<li>$\mathbf{w}^{(3)}$ : The office is about fifteen minutes from my house.</li>
<li>$\mathbf{w}^{(4)}$ : I want to learn how to play the guitar.</li>
<li>$\vdots$</li>
<li>$\mathbf{w}^{(i)}$ : <code>#include&lt;stdio.h&gt; int main(void){ printf("Hello, world!\n");}</code></li>
</ul>
<p>Assume $\mathbf{w}^{(i)} \overset{\text{i.i.d.}}{\sim} p_{\text{data}}$ and train $p_\theta$ to fit these samples.</p>

Note:
Code is text too; the same objective covers it. Source: Spring Lecture 02 slide 7, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/6.

---

<!-- .slide: id="chain-rule" -->

## Compute $P(w_{1:n})$

<ul>
<li>Write sentence $\mathbf{w}=[w_1,w_2,\ldots, w_n]$ as $w_{1:n}$: $P(\mathbf{w}) \equiv P(w_{1:n})$.</li>
<li>Let $X_1 = w_1, X_2 = w_2$; recall the chain rule $P(X_1X_2)=P(X_1)\cdot P(X_2\mid X_1)$.</li>
<li>Applying the chain rule to sentence $w_{1:n}$:</li>
</ul>
<p>$$P(w_{1:n})=\prod_{t=1}^{n} P\!\left(w_t \mid w_{1:t-1}\right)$$</p>
<ul>
<li>$w_{1:t-1}$ is called the <span class="text-good">history</span> of $w_t$. We assume $w_{1:0}=w_0=\text{BOS}$ and the last token $w_n=\text{EOS}$.</li>
<li>Computing $P(w_{1:n})$ is essentially a <span class="text-good">next word prediction problem</span>.</li>
</ul>

Note:
The chain rule is exact; no approximation yet. BOS and EOS make the first and last factors well defined. Source: Spring Lecture 02 slide 8, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/7.

---

<!-- .slide: id="next-token-prediction" -->

## Better next-token prediction

<blockquote><p class="text-bad">Key success of almost all modern LLMs: better next-token prediction $\Rightarrow$ better language modeling.</p></blockquote>

<video class="animation" src="assets/predicting-next-word.mp4" poster="assets/predicting-next-word-poster.jpg" controls playsinline preload="metadata" aria-label="Interview clip in which the speaker explains that large language models learn by predicting the next word."></video>

Note:
Play the clip (about two and a half minutes) or a part of it. The clip was bundled with the Spring deck without a stated source; see assets/README.md. Source: Spring Lecture 02 slide 8, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/7.

---

<!-- .slide: id="ngram-model" -->

## The $N$-gram model

<p><strong>Intuition:</strong> instead of using the entire history $w_{1:t-1}$, we can <span class="text-bad">approximate</span> the probability by using the last few ($N$) words.</p>
<ul>
<li><strong>Unigram model</strong> ($N=1$): approximates $P(\cdot)$ without history: $P(w_t \mid w_{1:t-1}) \approx p_\theta(w_t)$</li>
</ul>
<p>$$P(\text{skills}\mid \text{I want to improve my cooking}) \approx p_\theta(\text{skills})$$</p>
<ul>
<li><strong>Bigram model</strong> ($N=2$): approximates $P(\cdot)$ by only using $w_{t-1}$: $P(w_t \mid w_{1:t-1}) \approx p_\theta(w_t \mid w_{t-1})$</li>
</ul>
<p>$$P(\text{skills}\mid \text{I want to improve my cooking}) \approx p_\theta(\text{skills}\mid \text{cooking})$$</p>

Note:
Source: Spring Lecture 02 slide 9, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/8.

---

<!-- .slide: id="ngram-model-2" -->

## The $N$-gram model: Markov assumption

<ul>
<li><strong>Trigram model</strong> ($N=3$): approximates $P(w_t\mid w_{1:t-1})$ by only using $w_{t-2:t-1}$: $P(w_t\mid w_{1:t-1}) \approx p_\theta(w_t \mid w_{t-2:t-1})$</li>
</ul>
<p>$$P(\text{skills}\mid \text{I want to improve my cooking}) \approx p_\theta(\text{skills}\mid \text{my cooking})$$</p>
<p>The above approximations use the <strong>Markov assumption</strong>. In general, for $N$-gram ($N\ge 2$):</p>
<p>$$\text{(N-1)-order Markov:}\qquad P(w_t \mid w_{1:t-1}) \approx p_\theta(w_t \mid w_{t-N+1:t-1}).$$</p>

Note:
An $N$-gram model is an $(N-1)$-order Markov model over tokens. Source: Spring Lecture 02 slide 9, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/8.

---

<!-- .slide: id="build-ngram" -->

## Build N-gram LMs

<ul>
<li><strong>Step 1</strong>: Pick an order $N$.</li>
<li><strong>Step 2</strong>: Curate your training data $\mathcal{D}$ and vocabulary $\mathcal{V}$.
<ul><li>Step 2.1: Collect raw data (tokenization, filtering)</li><li>Step 2.2: Build vocabulary $\mathcal{V}$ after tokenization</li></ul></li>
<li><strong>Step 3</strong>: Estimate your parameters $p_\theta(w_t\mid w_{t-N+1:t-1})$. <strong>Q: How many parameters will we have?</strong></li>
<li><strong>Step 4</strong>: Test your model by generating sentences or estimating log probabilities (or perplexity) of given sentences in the test dataset.</li>
</ul>
<p class="caption"><span class="text-bad">Whiteboard:</span> estimate bigram $p_\theta(w_t\mid w_{t-1})$ (Exercise E01).</p>

Note:
Work the whiteboard derivation before showing the next slide: write the likelihood of the corpus under a bigram model and maximize it under the sum-to-one constraint for each history. Source: Spring Lecture 02 slide 10, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/9.

---

<!-- .slide: id="bigram-parameters" -->

## Bigram parameters and MLE

<p>Let $\mathcal{V}=\{v_1,v_2,\ldots,v_{|\mathcal{V}|}\}$. Define all possible parameters $\theta_{i,j} = p_\theta(w_t=v_i\mid w_{t-1}=v_j)$:</p>
<p>$$\boldsymbol{\theta} = \begin{bmatrix} \theta_{11} & \theta_{12} & \cdots & \theta_{1 |\mathcal{V}|}\\ \theta_{21} & \theta_{22} & \cdots & \theta_{2 |\mathcal{V}|}\\ \vdots & \vdots & \ddots & \vdots \\ \theta_{|\mathcal{V}| 1} & \theta_{|\mathcal{V}| 2} & \cdots & \theta_{|\mathcal{V}| |\mathcal{V}|} \end{bmatrix}$$</p>
<ul>
<li><strong>MLE for the bigram model:</strong> $p_\theta(v_i\mid v_j) = \dfrac{C(v_j\, v_i)}{C(v_j)}$, where $C(v_j\, v_i)$ counts the total frequency of the bigram $[v_j, v_i]$ in the training corpus.</li>
<li><strong>Potential parameters ($N=2$)</strong>: $\mathcal{O}(|\mathcal{V}|^2)$. <strong>$N$-gram is not scalable: $\mathcal{O}(|\mathcal{V}|^N)$.</strong></li>
</ul>

Note:
Each column of $\boldsymbol{\theta}$ is one conditional distribution and sums to one. The Spring slide wrote the MLE ratio as $C(v_i v_j)/C(v_i)$; the history $v_j$ belongs in the denominator, as written here. Source: Spring Lecture 02 slide 10, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/9.

---

<!-- .slide: class="exercise" id="exercise-01" -->

## Toy example of training a bigram LM

<p class="exercise-meta">Exercise E01 · 5 minutes · Notebook E01</p>

<p><strong>Maximum likelihood estimate</strong> (bigram): $p_\theta(w_i \mid w_{i-1}) = \frac{C(w_{i-1}w_i)}{\sum_{w\in \mathcal{V}} C(w_{i-1}w)} = \frac{C(w_{i-1}w_i)}{C(w_{i-1})}$, with $C(x)$ the frequency of $x$ in the corpus.</p>
<p>Training corpus, $\mathcal{V}=\{\text{EOS, I, am, Sam, do, not, like, eggs, and, ham}\}$:</p>
<blockquote><p><span class="text-token">BOS</span> <span class="text-good">I am Sam</span> <span class="text-token">EOS</span><br><span class="text-token">BOS</span> <span class="text-good">Sam I am</span> <span class="text-token">EOS</span><br><span class="text-token">BOS</span> <span class="text-good">I do not like eggs and ham</span> <span class="text-token">EOS</span></p></blockquote>

<div class="answer fragment"><p>$p_\theta(\text{I}\mid \text{BOS})=\frac{2}{3}=0.67$, $p_\theta(\text{Sam}\mid \text{BOS})=\frac{1}{3}=0.33$, $p_\theta(\text{am}\mid \text{I})=\frac{2}{3}=0.67$, $p_\theta(\text{do}\mid \text{I})=\frac{1}{3}=0.33$, $p_\theta(\text{EOS}\mid \text{Sam})=\frac{1}{2}=0.50$, $p_\theta(\text{Sam}\mid \text{am})=\frac{1}{2}=0.50$</p></div>

Note:
Ask students to compute the six estimates by hand, then reveal. Counts: $C(\text{BOS})=3$, $C(\text{BOS I})=2$, $C(\text{I})=3$, $C(\text{I am})=2$, $C(\text{Sam})=2$, $C(\text{Sam EOS})=1$, $C(\text{am})=2$, $C(\text{am Sam})=1$. Notebook E01 counts the same corpus in code. Source: Spring Lecture 02 slide 11, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/10.

---

<!-- .slide: id="restaurant-reviews" -->

## Bigram statistics from restaurant reviews

<ul>
<li>$\mathcal{D}$: restaurant reviews (e.g., 1000 sentences). Example sentences from reviews:
<ul><li>Wow... Loved this place ...</li><li>Not tasty and the texture was just nasty ...</li><li>The selection on the menu was great and ...</li></ul></li>
<li>Add boundary tokens: <span class="text-token">BOS</span> ... <span class="text-token">EOS</span></li>
<li>Bigram counts $C(w_{i-1},w_i)$; bigram MLE $p_\theta(w_i\mid w_{i-1})=\dfrac{C(w_{i-1},w_i)}{C(w_{i-1})}$</li>
<li>We can visualize a small sub-matrix for selected words (e.g., <span class="text-good">i, want, to, eat, in, this, place</span>).</li>
</ul>

Note:
The counts on the next slide come from the Spring notebook run on the sentiment-labelled-sentences dataset (3000 training rows). Source: Spring Lecture 02 slide 12, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/11.

---

<!-- .slide: id="restaurant-counts" -->

## Bigram counts (subset)

```text
split: train rows: 3000   columns: ['text']
Vocab size: 5166          Distinct bigrams: 22179

        i   want   to  eat  in  this  place
i       0     5     0    1   0     0      0
want    0     0    11    0   0     0      0
to      0     1     0   12   2     6      2
eat     0     0     0    0   2     0      0
in      0     0     3    0   0    24      2
this    1     0     4    0   2     0     73
place   2     0    10    0   3     0      0
```

Note:
Rows are the history word, columns the next word. Most cells are zero even for common words: this is the sparsity that smoothing addresses later. Source: Spring Lecture 02 slide 12, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/11.

---

<!-- .slide: id="practical-boundaries" -->

## Practical issues in $N$-gram LMs: sentence boundaries

<ul>
<li>Need extra context at the start and end.</li>
<li>For trigram: $p_\theta(w_1\mid \text{BOS},\text{BOS})$</li>
<li>Add tokens: <span class="text-token">BOS</span> … <span class="text-token">EOS</span></li>
</ul>

Note:
An $N$-gram model needs $N-1$ BOS tokens so that the first word has a full history. EOS lets the model assign probability to stopping, which makes the distribution over sentences of all lengths sum to one. Source: Spring Lecture 02 slide 13, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/12.

---

<!-- .slide: id="practical-oov" -->

## Practical issues in $N$-gram LMs: unknown words (OOV)

<p>A word like <em>Thisisahardtofindword</em> simply did not occur in our training set but could be in our test set.</p>
<ul>
<li><strong>Closed vocabulary:</strong> test words must be in a fixed lexicon.</li>
<li><strong>Open vocabulary:</strong> map unseen words to a pseudo-token <span class="text-token">&lt;UNK&gt;</span>.</li>
<li><strong>How to train</strong> $p_\theta(\text{UNK}\mid \cdot)$?
<ul><li><strong>Prior vocab:</strong> convert OOV in training to <span class="text-token">&lt;UNK&gt;</span>, then count it.</li><li><strong>No prior vocab:</strong> replace rare words (freq $\lt n$) with <span class="text-token">&lt;UNK&gt;</span> (or keep the top $|V|$ words).</li></ul></li>
</ul>

Note:
Connect to Lecture 01: subword tokenizers make OOV rare at the token level, but word-level $N$-gram models still need UNK. Source: Spring Lecture 02 slide 13, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/12.

---

<!-- .slide: class="outline-slide" id="outline-evaluation" -->

## Outline

<ul class="outline-topics">
<li>Probabilistic N-gram LMs</li>
<li aria-current="step">Evaluating LMs and Perplexity</li>
<li>Smoothing N-gram LMs</li>
<li>Neural Probabilistic LMs</li>
</ul>

Note:
Source: Spring Lecture 02 slide 14, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/13.

---

<!-- .slide: id="evaluation" -->

## Building LMs and evaluation

<p>Split your data into:</p>

| Training data | Validation data | Testing data |
| :--- | :--- | :--- |
| Estimate parameters | Tune choices | Report once |

<p><strong>Extrinsic evaluation</strong></p>
<ul>
<li>Compare models via <strong>downstream tasks</strong>. Examples: spell correction accuracy; machine translation accuracy.</li>
<li><span class="text-bad"><strong>Time-consuming</strong></span> (can take days or weeks).</li>
</ul>

Note:
The Spring slide shows the split as a proportional bar (about 70/15/15). The second table row is an added one-word summary of each part's role. Source: Spring Lecture 02 slide 15, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/14.

---

<!-- .slide: class="exercise" id="exercise-02" -->

## Intrinsic evaluation

<p class="exercise-meta">Exercise E02 · 3 minutes · Notebook E02</p>

<ul>
<li>Does the LM prefer <strong>good</strong> sentences to <strong>bad</strong> ones?</li>
<li>Train on the <strong>training dataset</strong>, evaluate on unseen <strong>test</strong> data. Never leak test sentences into training.</li>
<li>Probability-based metric: <strong>higher log-likelihood on test ⇒ better LM</strong>.</li>
</ul>
<p><span class="text-bad">Whiteboard:</span> propose a reasonable metric.</p>

<div class="answer fragment"><p>$$\frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{\mathbf{w}\in\mathcal{D}_{\text{test}}} \log p_\theta(\mathbf{w})$$</p></div>

Note:
Expected proposals: total log-likelihood, average per sentence, average per token. Discuss why per-token normalization makes different test sets comparable; this leads to perplexity. Source: Spring Lecture 02 slide 15, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/14.

---

<!-- .slide: id="perplexity" -->

## Perplexity

<p><strong>Definition (intrinsic metric)</strong></p>
<ul>
<li>A better LM assigns <strong>higher probability</strong> to an unseen test set $s_{1:m}$.</li>
<li><strong>Perplexity</strong> is the inverse probability of the test set, normalized by length $T=\sum_i |s_i|$:</li>
</ul>
<p>$$\mathrm{PPL}(s_{1:m}) = P_\theta(s_{1:m})^{-1/T} = \exp\!\left(-\frac{1}{T}\log P_\theta(s_{1:m})\right)$$</p>
<p>Minimizing PPL $\Longleftrightarrow$ maximizing test probability.</p>

Note:
The exponent of the average negative log-likelihood per token. With base-2 logarithms this is $2^{\text{cross-entropy}}$. Source: Spring Lecture 02 slide 16, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/15.

---

<!-- .slide: class="exercise" id="exercise-03" -->

## Perplexity: interpretation

<p class="exercise-meta">Exercise E03 · 3 minutes · Notebook E03</p>

<ul>
<li>Perplexity $\approx$ <strong>effective branching factor</strong>: how many plausible next tokens the model considers.</li>
<li>Example: random digits $\{0,\dots,9\}$, uniform guess $P=1/10$ for each digit. What is the perplexity of a digit string of length $t$?</li>
</ul>

<div class="answer fragment"><p>$$\mathrm{PPL}(s)=P(w_{1:t})^{-1/t} =\left(\left(\tfrac{1}{10}\right)^t\right)^{-1/t} =10$$</p>
<p>If the model is totally random, it needs $|\mathcal{V}|$ guesses on average to get the next word right. If $|\mathcal{V}|=1$, it surely picks the right one, hence $\mathrm{PPL}(s_{1:m})=1$.</p></div>

Note:
Expected answer: 10, independent of $t$. Notebook E03 computes it numerically for several lengths. Source: Spring Lecture 02 slide 16, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/15.

---

<!-- .slide: id="lower-perplexity" -->

## Lower perplexity – better model

Training 38 million words, test 1.5 million words (WSJ)

| N-gram order | Unigram | Bigram | Trigram |
| :--- | ---: | ---: | ---: |
| Perplexity | 962 | 170 | 109 |

- The improvement in perplexity does **not** guarantee an (extrinsic) improvement in downstream tasks like speech recognition or MT.
- Because perplexity often correlates with such improvements, it is commonly used as a **quick check** on an algorithm.

<p class="caption"><a href="https://arxiv.org/pdf/2005.14165.pdf" target="_blank" rel="noopener noreferrer">arxiv.org/pdf/2005.14165.pdf</a> (see how GPT-3 uses PPL) · <a href="https://nlpprogress.com/english/language_modeling.html" target="_blank" rel="noopener noreferrer">nlpprogress.com/english/language_modeling.html</a> (PPL on real datasets)</p>

Note:
The WSJ numbers are from Jurafsky and Martin, Chapter 3. Perplexities are only comparable across models that share the same vocabulary and tokenization. Source: Spring Lecture 02 slide 17, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/16.

---

<!-- .slide: id="sampling" -->

## Sentence sampling

<p><strong>Unigram case</strong></p>
<ul>
<li>Partition $[0,1]$ into intervals; each word gets an interval proportional to its frequency.</li>
<li>Sample $u\sim \mathrm{Unif}[0,1]$ and output the word whose interval contains $u$.</li>
<li>Repeat until we generate <span class="text-good">EOS</span>.</li>
</ul>
<p><strong>What about the bigram case?</strong></p>

Note:
Bigram answer: condition the interval partition on the previous word, starting from BOS. Notebook P01 samples from the toy bigram model. Source: Spring Lecture 02 slide 18, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/17.

---

<!-- .slide: id="sampling-figure" -->

## Unigram sampling on $[0,1]$

<img class="diagram" src="assets/sentence-sampling.png" alt="The unit interval from 0 to 1 divided into segments proportional to word probabilities: the 0.06, of 0.03, a 0.02, to 0.02, in 0.02, then many small segments, with however at 0.66 (p = 0.0003) and polyphonic near 0.99 (p = 0.0000018).">

<p class="caption">Frequent words own wide intervals; rare words own slivers near the end.</p>

Note:
Figure from Jurafsky and Martin, Chapter 3 (sampling from a unigram model), as used in the Spring deck. Source: Spring Lecture 02 slide 18, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/17.

---

<!-- .slide: id="sampling-outputs" -->

## Samples from a built LM

Trained on the Wall Street Journal (40M words):

- **1-gram:** Months the my and issue of year foreign new exchange's September were recession exchange new endorsed a acquire to six <span class="text-token">**executives**</span>
- **2-gram:** Last December through the way to preserve the <span class="text-token">**Hudson corporation**</span> ... would seem to complete the major central planners ... <span class="text-token">**M. X. corporation**</span> ...
- **3-gram:** They also point to ninety nine point <span class="text-token">**six billion dollars**</span> ... <span class="text-token">**six three percent**</span> ... <span class="text-token">**on market conditions**</span>

Note:
Higher order gives locally fluent phrases but still no global coherence. Samples from Jurafsky and Martin, Chapter 3. Source: Spring Lecture 02 slide 18, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/17.

---

<!-- .slide: class="outline-slide" id="outline-smoothing" -->

## Outline

<ul class="outline-topics">
<li>Probabilistic N-gram LMs</li>
<li>Evaluating LMs and Perplexity</li>
<li aria-current="step">Smoothing N-gram LMs</li>
<li>Neural Probabilistic LMs</li>
</ul>

Note:
Period 3 begins here. Source: Spring Lecture 02 slide 19, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/18.

---

<!-- .slide: id="smoothing" -->

## Smoothing N-gram LMs in one page

<p><strong>Problem:</strong> an unseen $n$-gram gives $q(\text{offer} \mid \text{denied the}) = 0$, so the sentence gets probability 0 and infinite perplexity. <strong>Fix:</strong> move a little mass from seen events to unseen ones.</p>
<p><strong>Add-$\delta$</strong> (Laplace when $\delta=1$; $|V|$ possible next words):</p>
<p>$$P_{\text{Add}}(w_i\mid w_{i-1}) =\frac{C(w_{i-1}w_i)+\delta}{C(w_{i-1})+\delta|V|}$$</p>
<p><strong>Linear interpolation</strong> ($\lambda_i$ tuned on held-out data, $\sum_i\lambda_i=1$):</p>
<p>$$P_{\text{Int}}(w_n\mid w_{n-2}w_{n-1}) = \lambda_1 P(w_n\mid w_{n-2}w_{n-1}) + \lambda_2 P(w_n\mid w_{n-1}) + \lambda_3 P(w_n)$$</p>
<p class="caption">Katz backoff and Kneser–Ney refine this idea (Jurafsky and Martin, Ch. 3). Neural LMs need no count smoothing.</p>

Note:
One page replaces the Spring section of nine slides. Add-one on the Berkeley Restaurant counts moves too much mass: $C(\text{i want})$ falls from 827 to a reconstituted 527 with $|V|=1446$; that is why $\delta \lt 1$ and interpolation are preferred. Notebook practices P02 (Laplace tables) and P03 (Good–Turing) keep the full worked examples for students who want them. Source: Spring Lecture 02 slides 20–27, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/19.

---

<!-- .slide: id="summary" -->

## Quick summary: N-gram language models

<ul>
<li><strong>Idea:</strong> approximate next-word probability using only the last $N-1$ words: $P(w_t \mid w_{1:t-1}) \approx q(w_t \mid w_{t-N+1:t-1})$</li>
<li><strong>Training:</strong> estimate counts from a corpus (plus smoothing for unseen $n$-grams).</li>
<li><strong>Two major issues</strong>
<ul><li><strong>Parameter explosion:</strong> the number of $n$-grams grows as $|V|^N$ (e.g., $|V|=10^4 \Rightarrow$ trigram $\sim 10^{12}$).</li><li><strong>Sparsity / poor generalization:</strong> many test $n$-grams never appear in training.</li></ul></li>
<li><strong>Today:</strong> neural LMs (RNN/Transformer) address these issues via learned representations.</li>
</ul>

Note:
Source: Spring Lecture 02 slide 28, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/27.

---

<!-- .slide: class="outline-slide" id="outline-nplm" -->

## Outline

<ul class="outline-topics">
<li>Probabilistic N-gram LMs</li>
<li>Evaluating LMs and Perplexity</li>
<li>Smoothing N-gram LMs</li>
<li aria-current="step">Neural Probabilistic LMs</li>
</ul>

Note:
Source: Spring Lecture 02 slide 29, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/28.

---

<!-- .slide: id="nplm-1" -->

## Neural Probabilistic LMs (NPLMs)

<img class="diagram" src="assets/nplm-1.png" alt="Left: language modeling calculates the probability of the next word; we have seen N-gram LMs; neural network LMs far outperform N-gram LMs, with Figure 1 of Bengio et al. (2003) showing table look-up in C, a tanh layer, and a softmax output. Right: NPLM is the same task as the N-gram model, predicting w_t from the last N-1 words, but represents words in the prior context by their embeddings; the header of the JMLR 2003 paper A Neural Probabilistic Language Model.">

<p class="caption">Bengio et al. (2003): NPLM. Same task as the $N$-gram model; words are represented by embeddings.</p>

Note:
The figure and text are the Spring slide image, kept unchanged. Source: Spring Lecture 02 slide 30, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/29.

---

<!-- .slide: id="nplm-2" -->

## Neural Probabilistic LMs (NPLMs)

<img class="diagram" src="assets/nplm-2.png" alt="Forward inference for a neural LM. Each of the N previous words is a one-hot vector of length |V|, multiplied by the embedding matrix E of size d by |V| to give a d-dimensional embedding. For P(w_t | w_{t-3:t-1}) at each timestep: concatenate the three embeddings into e, compute h = sigma(W e + b), z = U h, and y-hat = softmax(z). The diagram shows the words for, all, the feeding one-hot vectors through E, W, and U to output probabilities such as p(fish | ...).">

<p class="caption">Forward inference (decoding): $\mathbf{e}=[\mathbf{E}x_{t-3},\mathbf{E}x_{t-2},\mathbf{E}x_{t-1}]$, $\mathbf{h}=\sigma(\mathbf{W}\mathbf{e}+\mathbf{b})$, $\hat{\mathbf{y}}=\text{softmax}(\mathbf{U}\mathbf{h})$.</p>

Note:
Diagram from Jurafsky and Martin, Chapter 7. Source: Spring Lecture 02 slide 31, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/30.

---

<!-- .slide: id="nplm-3" -->

## Neural Probabilistic LMs (NPLMs)

<img class="diagram" src="assets/nplm-3.png" alt="Training the neural LM: learn the embeddings simultaneously with training the network; useful when the task the network is designed for is text classification, translation, or parsing. The diagram highlights the embedding matrix E as model parameters and the loss L = minus log P(fish | for, all, the) for the target word fish.">

<p class="caption">Embeddings as model parameters, learned with the loss $L=-\log P(\text{fish}\mid\text{for, all, the})$.</p>

Note:
Diagram from Jurafsky and Martin, Chapter 7. Source: Spring Lecture 02 slide 32, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/31.

---

<!-- .slide: id="nplm-4" -->

## Neural Probabilistic LMs (NPLMs)

<img class="diagram" src="assets/nplm-4.png" alt="Improvements over N-gram LM: tackles the sparsity problem; model size is relatively small, O(d times |V|), compared with O(|V| to the N) where N is the window size. Table of perplexities on the AP News corpus: MLP10 (n = 6) valid 104, test 109; deleted interpolation (n = 3) 126 and 132; back-off Kneser-Ney with n = 3, 4, 5 gives test 127, 119, 117. Right: a diagram with context words and, our, problems, turning in a window of size 4, lookup embeddings, concatenate, W1, a hidden layer, W2, softmax, and a probability bar chart over mat, table, bed, desk, chair for the target word into.">

<p class="caption">MLP10 = NPLM. Comparative results on the AP News corpus (Bengio et al., 2003).</p>

Note:
NPLM already beat the best smoothed $N$-gram models in 2003; the gap widened with RNNs and Transformers. Source: Spring Lecture 02 slide 33, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/32.

---

<!-- .slide: class="references" id="references" -->

## Language model toolkits and readings

- **Reading:** [Jurafsky and Martin, *N-gram Language Models*](https://web.stanford.edu/~jurafsky/slp3/3.pdf)<br>Chain rule, MLE, perplexity, smoothing.
- **KenLM** (fast $n$-gram LM toolkit): <a href="https://kheafield.com/code/kenlm/" target="_blank" rel="noopener noreferrer">kheafield.com/code/kenlm/</a>
- **Google N-Gram Release (Aug 2006):** <a href="https://ai.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html" target="_blank" rel="noopener noreferrer">ai.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html</a><br>Tokens: 1,024,908,267,229 · Sentences: 95,119,665,584 · Unigrams: 13,588,391 · Fivegrams: 1,176,470,663
- **SRILM** (classic LM toolkit): <a href="http://www.speech.sri.com/projects/srilm/" target="_blank" rel="noopener noreferrer">speech.sri.com/projects/srilm/</a>
- **Alias method** (fast discrete sampling): <a href="https://www.keithschwarz.com/darts-dice-coins/" target="_blank" rel="noopener noreferrer">keithschwarz.com/darts-dice-coins/</a>

Note:
The Spring slide closed with the reading for its next lecture (Chapters 4–5: Naive Bayes, logistic regression, embeddings). In the Fall sequence, Chapter 3 is the reading for this lecture; Week 3 continues with embeddings. Source: Spring Lecture 02 slide 34, https://baojian.github.io/llm-26/slides/lecture-02-slides/index.html#/33.
