<!-- .slide: id="source-001" class="native-slide" -->

<div class="native-text n001" data-source-text-id="s001-text2">
<h1 class="native-paragraph n003"><span class="n002">Lecture 11 – Diffusion Language Models</span></h1>
</div>
<div class="native-text n004" data-source-text-id="s001-text3">
<p class="native-paragraph n006"><span class="n005">Baojian Zhou</span></p>
<p class="native-paragraph n006"><span class="n005">DATA130030.01 (自然语言处理)</span></p>
<p class="native-paragraph n006"><span class="n005">School of Data Science Fudan University</span></p>
<p class="native-paragraph n006"><span class="n005">11/18/2026</span></p>
</div>
<div class="native-text n007" data-source-text-id="s001-text8">
<p class="native-paragraph n009"><span class="n008">1</span></p>
</div>

---

<!-- .slide: id="source-002" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n011" data-source-text-id="s002-text3">
<p class="native-paragraph n009"><span class="n008">2</span></p>
</div>
<div class="native-text n012" data-source-text-id="s002-text4">
<h2 class="native-paragraph n014"><span class="n013">Introduction</span></h2>
</div>

---

<!-- .slide: id="source-003" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s003-text2">
<h2 class="native-paragraph n017"><span class="n016">What is diffusion model?</span></h2>
</div>
<div class="native-text n018" data-source-text-id="s003-text3">
<p class="native-paragraph n020"><span class="n019">Creating noise from data is easy; creating data from noise is generative modeling</span></p>
</div>
<div class="native-text n011" data-source-text-id="s003-text4">
<p class="native-paragraph n009"><span class="n008">3</span></p>
</div>
<div class="native-text n021" data-source-text-id="s003-text6">
<p class="native-paragraph n023"><span class="n022"><a href="https://arxiv.org/pdf/2011.13456">Song, Yang, et al. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456 (2020).</a></span></p>
</div>

---

<!-- .slide: id="source-004" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s004-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s004-text4">
<p class="native-paragraph n009"><span class="n008">4</span></p>
</div>
<div id="s004-pic2050" class="native-figure n025"><img class="n024" data-src="assets/media/image1.jpeg" alt="Diffusion models in image world — original illustration 2050"></div>
<div class="native-text n026" data-source-text-id="s004-text6">
<p class="native-paragraph n028 native-bullet"><span class="n027">Théâtre D&#x27;opéra Spatial</span></p>
</div>
<div class="native-text n029" data-source-text-id="s004-text8">
<p class="native-paragraph n031"><span class="n030">A digital arts piece generated and edited by an American man named Jason M. Allen with AI model Midjourney</span></p>
</div>

---

<!-- .slide: id="source-005" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s005-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div id="s005-pic5" class="native-figure n033"><img class="n032" data-src="assets/media/image2.png" alt="Diffusion models in image world — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s005-text4">
<p class="native-paragraph n009"><span class="n008">5</span></p>
</div>

---

<!-- .slide: id="source-006" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s006-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div id="s006-pic5" class="native-figure n034"><img class="n032" data-src="assets/media/image3.png" alt="Diffusion models in image world — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s006-text4">
<p class="native-paragraph n009"><span class="n008">6</span></p>
</div>
<div id="s006-pic6" class="native-figure n036 fragment"><img class="n035" data-src="assets/media/image4.png" alt="Diffusion models in image world — original illustration 6"></div>

---

<!-- .slide: id="source-007" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s007-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div id="s007-pic5" class="native-figure n037"><img class="n032" data-src="assets/media/image5.png" alt="Diffusion models in image world — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s007-text4">
<p class="native-paragraph n009"><span class="n008">7</span></p>
</div>

---

<!-- .slide: id="source-008" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s008-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s008-text4">
<p class="native-paragraph n009"><span class="n008">8</span></p>
</div>
<div id="s008-pic7" class="native-figure n038"><img class="n032" data-src="assets/media/image6.png" alt="Diffusion models in image world — original illustration 7"></div>

---

<!-- .slide: id="source-009" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s009-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s009-text4">
<p class="native-paragraph n009"><span class="n008">9</span></p>
</div>
<div id="s009-pic7" class="native-figure n039"><img class="n032" data-src="assets/media/image7.png" alt="Diffusion models in image world — original illustration 7"></div>

---

<!-- .slide: id="source-010" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s010-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s010-text4">
<p class="native-paragraph n009"><span class="n008">10</span></p>
</div>
<div id="s010-pic7" class="native-figure n040"><img class="n032" data-src="assets/media/image8.png" alt="Diffusion models in image world — original illustration 7"></div>

---

<!-- .slide: id="source-011" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s011-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s011-text4">
<p class="native-paragraph n009"><span class="n008">11</span></p>
</div>
<div id="s011-pic7" class="native-figure n041"><img class="n032" data-src="assets/media/image9.png" alt="Diffusion models in image world — original illustration 7"></div>

---

<!-- .slide: id="source-012" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s012-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s012-text4">
<p class="native-paragraph n009"><span class="n008">12</span></p>
</div>
<div id="s012-pic7" class="native-figure n042"><img class="n032" data-src="assets/media/image10.png" alt="Diffusion models in image world — original illustration 7"></div>

---

<!-- .slide: id="source-013" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s013-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s013-text4">
<p class="native-paragraph n009"><span class="n008">13</span></p>
</div>
<div id="s013-pic7" class="native-figure n043"><img class="n032" data-src="assets/media/image11.png" alt="Diffusion models in image world — original illustration 7"></div>

---

<!-- .slide: id="source-014" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s014-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div id="s014-pic5" class="native-figure n044"><img class="n032" data-src="assets/media/image12.png" alt="Diffusion models in image world — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s014-text4">
<p class="native-paragraph n009"><span class="n008">14</span></p>
</div>

---

<!-- .slide: id="source-015" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s015-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div id="s015-pic5" class="native-figure n045"><img class="n032" data-src="assets/media/image13.png" alt="Diffusion models in image world — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s015-text4">
<p class="native-paragraph n009"><span class="n008">15</span></p>
</div>

---

<!-- .slide: id="source-016" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s016-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">training</span></h2>
</div>
<div id="s016-pic5" class="native-figure n047"><img class="n032" data-src="assets/media/image14.png" alt="Diffusion models in image world training — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s016-text4">
<p class="native-paragraph n009"><span class="n008">16</span></p>
</div>

---

<!-- .slide: id="source-017" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s017-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">training</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s017-text4">
<p class="native-paragraph n009"><span class="n008">17</span></p>
</div>
<div id="s017-pic7" class="native-figure n048"><img class="n032" data-src="assets/media/image15.png" alt="Diffusion models in image world training — original illustration 7"></div>

---

<!-- .slide: id="source-018" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s018-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">training</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s018-text4">
<p class="native-paragraph n009"><span class="n008">18</span></p>
</div>
<div id="s018-pic7" class="native-figure n049"><img class="n032" data-src="assets/media/image16.png" alt="Diffusion models in image world training — original illustration 7"></div>

---

<!-- .slide: id="source-019" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s019-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">training</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s019-text4">
<p class="native-paragraph n009"><span class="n008">19</span></p>
</div>
<div id="s019-pic7" class="native-figure n050"><img class="n032" data-src="assets/media/image17.png" alt="Diffusion models in image world training — original illustration 7"></div>

---

<!-- .slide: id="source-020" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s020-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">training</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s020-text4">
<p class="native-paragraph n009"><span class="n008">20</span></p>
</div>
<div id="s020-pic7" class="native-figure n051"><img class="n032" data-src="assets/media/image18.png" alt="Diffusion models in image world training — original illustration 7"></div>

---

<!-- .slide: id="source-021" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s021-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">training</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s021-text4">
<p class="native-paragraph n009"><span class="n008">21</span></p>
</div>
<div id="s021-pic7" class="native-figure n052"><img class="n032" data-src="assets/media/image19.png" alt="Diffusion models in image world training — original illustration 7"></div>

---

<!-- .slide: id="source-022" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s022-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">training</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s022-text4">
<p class="native-paragraph n009"><span class="n008">22</span></p>
</div>
<div id="s022-pic7" class="native-figure n053"><img class="n032" data-src="assets/media/image20.png" alt="Diffusion models in image world training — original illustration 7"></div>

---

<!-- .slide: id="source-023" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s023-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s023-text4">
<p class="native-paragraph n009"><span class="n008">23</span></p>
</div>
<div id="s023-pic7" class="native-figure n054"><img class="n032" data-src="assets/media/image21.png" alt="Diffusion models in image world — original illustration 7"></div>

---

<!-- .slide: id="source-024" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s024-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">sampling</span></h2>
</div>
<div id="s024-pic5" class="native-figure n055"><img class="n032" data-src="assets/media/image22.png" alt="Diffusion models in image world sampling — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s024-text4">
<p class="native-paragraph n009"><span class="n008">24</span></p>
</div>

---

<!-- .slide: id="source-025" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s025-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">sampling</span></h2>
</div>
<div id="s025-pic5" class="native-figure n056"><img class="n032" data-src="assets/media/image23.png" alt="Diffusion models in image world sampling — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s025-text4">
<p class="native-paragraph n009"><span class="n008">25</span></p>
</div>

---

<!-- .slide: id="source-026" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s026-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">sampling</span></h2>
</div>
<div id="s026-pic5" class="native-figure n057"><img class="n032" data-src="assets/media/image24.png" alt="Diffusion models in image world sampling — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s026-text4">
<p class="native-paragraph n009"><span class="n008">26</span></p>
</div>

---

<!-- .slide: id="source-027" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s027-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">sampling</span></h2>
</div>
<div id="s027-pic5" class="native-figure n058"><img class="n032" data-src="assets/media/image25.png" alt="Diffusion models in image world sampling — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s027-text4">
<p class="native-paragraph n009"><span class="n008">27</span></p>
</div>

---

<!-- .slide: id="source-028" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s028-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">sampling</span></h2>
</div>
<div id="s028-pic5" class="native-figure n059"><img class="n032" data-src="assets/media/image26.png" alt="Diffusion models in image world sampling — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s028-text4">
<p class="native-paragraph n009"><span class="n008">28</span></p>
</div>

---

<!-- .slide: id="source-029" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s029-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world </span><span class="n046">sampling</span></h2>
</div>
<div id="s029-pic5" class="native-figure n060"><img class="n032" data-src="assets/media/image27.png" alt="Diffusion models in image world sampling — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s029-text4">
<p class="native-paragraph n009"><span class="n008">29</span></p>
</div>

---

<!-- .slide: id="source-030" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s030-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in image world</span></h2>
</div>
<div id="s030-pic5" class="native-figure n061"><img class="n032" data-src="assets/media/image28.png" alt="Diffusion models in image world — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s030-text4">
<p class="native-paragraph n009"><span class="n008">30</span></p>
</div>

---

<!-- .slide: id="source-031" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s031-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in language</span></h2>
</div>
<div class="native-text n062" data-source-text-id="s031-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">A text generation method different from the traditional left-to-right approach</span></p>
</div>
<div class="native-text n011" data-source-text-id="s031-text4">
<p class="native-paragraph n009"><span class="n008">31</span></p>
</div>
<div id="s031-pic8" class="native-figure n065"><img class="n024 native-gif" src="assets/media/image29-poster.png" data-gif-source="assets/media/image29.gif" data-poster="assets/media/image29-poster.png" alt="Diffusion models in language — original illustration 8"></div>
<div class="native-text n066" data-source-text-id="s031-text9">
<p class="native-paragraph n068"><span class="n067"><a href="https://ml-gsai.github.io/LLaDA-demo/">LLaDA model</a></span></p>
</div>

---

<!-- .slide: id="source-032" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s032-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in language</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s032-text4">
<p class="native-paragraph n009"><span class="n008">32</span></p>
</div>
<div class="native-text n069" data-source-text-id="s032-text5">
<p class="native-paragraph n064 native-bullet"><span class="n063">LLaDA demonstrates impressive scalability, with its overall trend being highly competitive with that of AR baseline on the same data.</span></p>
</div>
<div class="native-text n066" data-source-text-id="s032-text6">
<p class="native-paragraph n068"><span class="n067"><a href="https://ml-gsai.github.io/LLaDA-demo/">LLaDA model</a></span></p>
</div>
<div id="s032-pic9" class="native-figure n070"><img class="n032" data-src="assets/media/image30.png" alt="Diffusion models in language — original illustration 9"></div>

---

<!-- .slide: id="source-033" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s033-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in language</span></h2>
</div>
<div class="native-text n071" data-source-text-id="s033-text3">
<p class="native-paragraph n073 native-bullet"><span class="n072">Diffusion models are not constrained to sequential (e.g., left-to-right) generation, enabling outputs to be synthesized in arbitrary orders. This allows for more diverse user queries.</span></p>
</div>
<div class="native-text n011" data-source-text-id="s033-text4">
<p class="native-paragraph n009"><span class="n008">33</span></p>
</div>
<div id="s033-pic3074" class="native-figure n074"><img class="n024 native-gif" src="assets/media/image31-poster.png" data-gif-source="assets/media/image31.gif" data-poster="assets/media/image31-poster.png" alt="Diffusion models in language — original illustration 3074"></div>
<div class="native-text n075" data-source-text-id="s033-text5">
<p class="native-paragraph n068"><span class="n067"><a href="https://hkunlp.github.io/blog/2025/dream/">Dream model</a></span></p>
</div>

---

<!-- .slide: id="source-034" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s034-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in language</span></h2>
</div>
<div class="native-text n076" data-source-text-id="s034-text3">
<p class="native-paragraph n078 native-bullet"><span class="n077">Controlling the decoding behavior</span></p>
</div>
<div class="native-text n011" data-source-text-id="s034-text4">
<p class="native-paragraph n009"><span class="n008">34</span></p>
</div>
<div class="native-text n075" data-source-text-id="s034-text5">
<p class="native-paragraph n068"><span class="n067"><a href="https://hkunlp.github.io/blog/2025/dream/">Dream model</a></span></p>
</div>
<div id="s034-pic6" class="native-figure n079"><img class="n024 native-gif" src="assets/media/image32-poster.png" data-gif-source="assets/media/image32.gif" data-poster="assets/media/image32-poster.png" alt="Diffusion models in language — original illustration 6"></div>
<div id="s034-pic5122" class="native-figure n080"><img class="n024 native-gif" src="assets/media/image33-poster.png" data-gif-source="assets/media/image33.gif" data-poster="assets/media/image33-poster.png" alt="Diffusion models in language — original illustration 5122"></div>
<div id="s034-pic5124" class="native-figure n081"><img class="n024 native-gif" src="assets/media/image34-poster.png" data-gif-source="assets/media/image34.gif" data-poster="assets/media/image34-poster.png" alt="Diffusion models in language — original illustration 5124"></div>

---

<!-- .slide: id="source-035" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s035-text2">
<h2 class="native-paragraph n017"><span class="n016">Diffusion models in language</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s035-text4">
<p class="native-paragraph n009"><span class="n008">35</span></p>
</div>
<div class="native-text n082" data-source-text-id="s035-text5">
<p class="native-paragraph n084 native-bullet"><span class="n083">Quality-speed Trade-off</span></p>
</div>
<div id="s035-pic4100" class="native-figure n085"><img class="n024" data-src="assets/media/image35.png" alt="Diffusion models in language — original illustration 4100"></div>
<div class="native-text n075" data-source-text-id="s035-text6">
<p class="native-paragraph n068"><span class="n067"><a href="https://hkunlp.github.io/blog/2025/dream/">Dream model</a></span></p>
</div>

---

<!-- .slide: id="source-036" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n011" data-source-text-id="s036-text3">
<p class="native-paragraph n009"><span class="n008">36</span></p>
</div>
<div class="native-text n012" data-source-text-id="s036-text4">
<h2 class="native-paragraph n014"><span class="n013">Diffusion Models</span></h2>
</div>

---

<!-- .slide: id="source-037" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s037-text2">
<h2 class="native-paragraph n017"><span class="n016">Autoencoder</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s037-text4">
<p class="native-paragraph n009"><span class="n008">37</span></p>
</div>
<div id="s037-pic1026" class="native-figure n086"><img class="n024" data-src="assets/media/image36.png" alt="Autoencoder — original illustration 1026"></div>
<div class="native-text n087" data-source-text-id="s037-text5">
<p class="native-paragraph n064 native-bullet"><span class="n063">Autoencoder is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input.</span></p>
</div>
<div class="native-text n088" data-source-text-id="s037-text7">
<p class="native-paragraph n068"><span class="n067"><a href="https://www.cs.toronto.edu/~hinton/absps/science.pdf">Reducing the Dimensionality of Data with Neural Networks</a></span></p>
</div>
<div class="native-text n089" data-source-text-id="s037-text9">
<p class="native-paragraph n090 native-bullet"><span class="n063">The encoder network essentially accomplishes the dimensionality reduction, just like we use Principal Component Analysis (PCA).</span></p>
</div>

---

<!-- .slide: id="source-038" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s038-text2">
<h2 class="native-paragraph n017"><span class="n016">Autoencoder</span></h2>
</div>
<div class="native-text n091" data-source-text-id="s038-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">The model contains an encoder function </span><span class="native-math">$g(\cdot )$</span><span class="n063"> parameterized by </span><span class="native-math">$\phi $</span><span class="n063"> and a decoder function </span><span class="native-math">$f(\cdot )$</span><span class="n063"> parameterized by </span><span class="native-math">$\boldsymbol{\theta }$</span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">The low-dimensional code learned for input </span><span class="native-math">$\boldsymbol{x}$</span><span class="n063"> in the bottleneck layer is </span><span class="native-math">$\boldsymbol{z}={g}_{\phi }\left(\boldsymbol{x}\right)$</span><span class="n063"> and the reconstructed input is </span><span class="native-math">${\boldsymbol{x} }^{\prime }={f}_{\theta }({g}_{\phi }(\boldsymbol{x}))$</span></p>
<p class="native-paragraph n094 native-bullet"><span class="n093">The parameters </span><span class="native-math">$\textcolor{#1F1F1F}{(}\textcolor{#1F1F1F}{\boldsymbol{\phi } }\textcolor{#1F1F1F}{\boldsymbol{,} }\textcolor{#1F1F1F}{\boldsymbol{\theta } }\textcolor{#1F1F1F}{)}$</span><span class="n093"> are learned together to output a reconstructed data sample same as the original input</span></p>
</div>
<div class="native-text n011" data-source-text-id="s038-text4">
<p class="native-paragraph n009"><span class="n008">38</span></p>
</div>
<div id="s038-pic5" class="native-figure n095"><img class="n032" data-src="assets/media/image38.png" alt="Autoencoder — original illustration 5"></div>

---

<!-- .slide: id="source-039" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s039-text2">
<h2 class="native-paragraph n017"><span class="n016">Autoencoder </span><span class="n046">denoising version</span></h2>
</div>
<div class="native-text n096" data-source-text-id="s039-text3">
<p class="native-paragraph n098 native-bullet"><span class="n097">The standard autoencoder quantifies the difference between two vectors, such as cross entropy when the activation function is sigmoid, or as simple as MSE loss:</span></p>
<p class="native-paragraph n099 native-display-math"><span class="native-math">$\displaystyle {L}_{AE}\left(\theta ,\phi \right)=\frac{1}{n}\sum\limits_{i=1}^{n}{ {\left({\boldsymbol{x} }^{(i)}-{f}_{\theta }({g}_{\phi }({\boldsymbol{x} }^{(i)}))\right)}^{2} }$</span></p>
<p class="native-paragraph n100 native-bullet"><span class="n097">The denoising version tries to avoid overfitting and improve the robustness</span></p>
<p class="native-paragraph n099 native-display-math"><span class="native-math">$\displaystyle {L}_{\textcolor{#00B050}{DAE} }\left(\theta ,\phi \right)=\frac{1}{n}\sum\limits_{i=1}^{n}{ {\left({\boldsymbol{x} }^{(i)}-{f}_{\theta }({g}_{\phi }({\textcolor{#00B050}{\tilde{\textcolor{#00B050}{\boldsymbol{x} } } } }^{(i)}))\right)}^{2} }$</span></p>
</div>
<div class="native-text n011" data-source-text-id="s039-text4">
<p class="native-paragraph n009"><span class="n008">39</span></p>
</div>
<div id="s039-pic8196" class="native-figure n101"><img class="n024" data-src="assets/media/image40.png" alt="Autoencoder denoising version — original illustration 8196"></div>
<div id="s039-pic5" class="native-figure n102"><img class="n024" data-src="assets/media/image36.png" alt="Autoencoder denoising version — original illustration 5"></div>
<div aria-hidden="true" class="native-decoration n103"></div>
<div aria-hidden="true" class="native-decoration n104"></div>

---

<!-- .slide: id="source-040" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s040-text2">
<h2 class="native-paragraph n017"><span class="n016">Variational Autoencoder (VAE)</span></h2>
</div>
<div class="native-text n105" data-source-text-id="s040-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Autoencoders</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">learn a low dimensional representation of the input, but hard to work with as a generative model</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">one of the key limitations of autoencoders is that we have no way of sampling from them!</span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">Variational autoencoders (VAEs)</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">by contrast learn a continuous latent space that is easy to sample from</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">can generate new data (e.g. images) by sampling from the learned generative model</span></p>
</div>
<div class="native-text n011" data-source-text-id="s040-text4">
<p class="native-paragraph n009"><span class="n008">40</span></p>
</div>
<div id="s040-pic9218" class="native-figure n107"><img class="n024" data-src="assets/media/image41.png" alt="Variational Autoencoder (VAE) — original illustration 9218"></div>
<div id="s040-pic5" class="native-figure n108"><img class="n024" data-src="assets/media/image36.png" alt="Variational Autoencoder (VAE) — original illustration 5"></div>

---

<!-- .slide: id="source-041" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s041-text2">
<h2 class="native-paragraph n017"><span class="n016">Variational Autoencoder (VAE)</span></h2>
</div>
<div class="native-text n109" data-source-text-id="s041-text3">
<p class="native-paragraph n111 native-bullet"><span class="n110">Neural Network Perspective</span></p>
<p class="native-paragraph n113 native-bullet"><span class="n112">We can view a variational autoencoder (VAE) as an autoencoder consisting of two neural networks</span></p>
<p class="native-paragraph n113 native-bullet"><span class="n112">VAEs (as autoencoders) define two distributions</span></p>
<p class="native-paragraph n115 native-bullet"><span class="n114">encoder: </span><span class="native-math">${q}_{\boldsymbol{\phi } }\left(\boldsymbol{z}\middle|\boldsymbol{x}\right)$</span></p>
<p class="native-paragraph n115 native-bullet"><span class="n114">decoder: </span><span class="native-math">${p}_{\boldsymbol{\theta } }\left(\boldsymbol{x}\middle|\boldsymbol{z}\right)$</span></p>
<p class="native-paragraph n116 native-bullet"><span class="n110">We can also view the VAE from the perspective of variational inference</span></p>
<p class="native-paragraph n113 native-bullet"><span class="n112">Model: </span><span class="native-math">${q}_{\boldsymbol{\phi } }\left(\boldsymbol{z}\middle|\boldsymbol{x}\right)$</span></p>
<p class="native-paragraph n113 native-bullet"><span class="n112">Variational approximation: </span><span class="native-math">${p}_{\boldsymbol{\theta } }\left(\boldsymbol{x}\middle|\boldsymbol{z}\right)$</span></p>
</div>
<div class="native-text n011" data-source-text-id="s041-text4">
<p class="native-paragraph n009"><span class="n008">41</span></p>
</div>
<div id="s041-pic5" class="native-figure n117"><img class="n032" data-src="assets/media/image43.png" alt="Variational Autoencoder (VAE) — original illustration 5"></div>
<div id="s041-pic6" class="native-figure n118"><img class="n024" data-src="assets/media/image41.png" alt="Variational Autoencoder (VAE) — original illustration 6"></div>

---

<!-- .slide: id="source-042" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s042-text2">
<h2 class="native-paragraph n017"><span class="n016">Variational Autoencoder (VAE)</span></h2>
</div>
<div class="native-text n119" data-source-text-id="s042-text3">
<p class="native-paragraph n121 native-bullet"><span class="n120">VAEs for Image Generation</span></p>
</div>
<div class="native-text n011" data-source-text-id="s042-text4">
<p class="native-paragraph n009"><span class="n008">42</span></p>
</div>
<div id="s042-pic5" class="native-figure n122"><img class="n032" data-src="assets/media/image44.png" alt="Variational Autoencoder (VAE) — original illustration 5"></div>
<div id="s042-pic6" class="native-figure n123"><img class="n032" data-src="assets/media/image45.png" alt="Variational Autoencoder (VAE) — original illustration 6"></div>

---

<!-- .slide: id="source-043" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s043-text2">
<h2 class="native-paragraph n017"><span class="n016">Variational Autoencoder (VAE)</span></h2>
</div>
<div class="native-text n124" data-source-text-id="s043-text3">
<p class="native-paragraph n121 native-bullet"><span class="n120">Random samples from learned generative models of MNIST for different dimensionalities of latent space</span></p>
</div>
<div class="native-text n011" data-source-text-id="s043-text4">
<p class="native-paragraph n009"><span class="n008">43</span></p>
</div>
<div id="s043-pic5" class="native-figure n125"><img class="n032" data-src="assets/media/image46.png" alt="Variational Autoencoder (VAE) — original illustration 5"></div>

---

<!-- .slide: id="source-044" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s044-text2">
<h2 class="native-paragraph n017"><span class="n016">Discrete VAE</span></h2>
</div>
<div class="native-text n126" data-source-text-id="s044-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">example of an application of VAEs to discrete data</span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">built on the sequence-to-sequence framework</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">input is read in by an LSTM</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">output is generated by an LSTM-LM</span></p>
</div>
<div class="native-text n011" data-source-text-id="s044-text4">
<p class="native-paragraph n009"><span class="n008">44</span></p>
</div>
<div id="s044-pic5" class="native-figure n127"><img class="n032" data-src="assets/media/image47.png" alt="Discrete VAE — original illustration 5"><a class="native-picture-link" href="https://aclanthology.org/K16-1002.pdf" aria-label="https://aclanthology.org/K16-1002.pdf"></a></div>
<div id="s044-pic6" class="native-figure n128"><img class="n032" data-src="assets/media/image48.png" alt="Discrete VAE — original illustration 6"></div>

---

<!-- .slide: id="source-045" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s045-text2">
<h2 class="native-paragraph n017"><span class="n016">Discrete VAE</span></h2>
</div>
<div class="native-text n076" data-source-text-id="s045-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">VAEs for Text Generation: Sentences produced by greedily decoding from points between two sentence encodings with a conventional autoencoder. The intermediate sentences are not plausible English</span></p>
</div>
<div class="native-text n011" data-source-text-id="s045-text4">
<p class="native-paragraph n009"><span class="n008">45</span></p>
</div>
<div id="s045-pic5" class="native-figure n129"><img class="n032" data-src="assets/media/image49.png" alt="Discrete VAE — original illustration 5"></div>

---

<!-- .slide: id="source-046" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s046-text2">
<h2 class="native-paragraph n017"><span class="n016">Discrete VAE</span></h2>
</div>
<div class="native-text n130" data-source-text-id="s046-text3">
<p class="native-paragraph n121 native-bullet"><span class="n120">Examples of using beam search to impute missing words within sentences.</span></p>
</div>
<div class="native-text n011" data-source-text-id="s046-text4">
<p class="native-paragraph n009"><span class="n008">46</span></p>
</div>
<div id="s046-pic5" class="native-figure n131"><img class="n032" data-src="assets/media/image50.png" alt="Discrete VAE — original illustration 5"></div>

---

<!-- .slide: id="source-047" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s047-text2">
<h2 class="native-paragraph n017"><span class="n016">VQ-VAE</span></h2>
</div>
<div class="native-text n132" data-source-text-id="s047-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Vector Quantized VAE: model learns a discrete latent variable by the encoder, since discrete representations may be a more natural fit for problems like language, speech, reasoning, etc.</span></p>
</div>
<div class="native-text n011" data-source-text-id="s047-text4">
<p class="native-paragraph n009"><span class="n008">47</span></p>
</div>
<div id="s047-pic10242" class="native-figure n133"><img class="n024" data-src="assets/media/image51.png" alt="VQ-VAE — original illustration 10242"></div>

---

<!-- .slide: id="source-048" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s048-text2">
<h2 class="native-paragraph n017"><span class="n016">VQ-VAE</span></h2>
</div>
<div class="native-text n076" data-source-text-id="s048-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063"> latent embedding space (also known as “codebook”): </span><span class="native-math">$\boldsymbol{e}\in {R}^{K\times D}$</span></p>
<p class="native-paragraph n106 native-bullet"><span class="native-math">$K$</span><span class="n097"> is the number of latent variable categories</span></p>
<p class="native-paragraph n106 native-bullet"><span class="native-math">$D$</span><span class="n097"> is the embedding size. An individual embedding vector is </span><span class="native-math">${\boldsymbol{e} }_{i}\in {R}^{D}$</span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">Encoder </span><span class="native-math">$E(\boldsymbol{x})$</span><span class="n063"> output  goes through a nearest-neighbor lookup to match to one of </span><span class="native-math">$K$</span><span class="n063"> embedding vectors </span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">matched code vector becomes the input for the decoder </span><span class="native-math">$D(\cdot )$</span></p>
</div>
<div class="native-text n011" data-source-text-id="s048-text4">
<p class="native-paragraph n009"><span class="n008">48</span></p>
</div>

---

<!-- .slide: id="source-049" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s049-text2">
<h2 class="native-paragraph n017"><span class="n016">DDPMs</span></h2>
</div>
<div class="native-text n134" data-source-text-id="s049-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Denoising Diffusion Probabilistic Models (DDPMs): a class of likelihood-based models that generate samples by gradually removing noise from a signal.</span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">Given a dataset of samples </span><span class="native-math">$\boldsymbol{x}$</span><span class="n063"> drawn from </span><span class="native-math">${p}_{\mathrm{data} }(\boldsymbol{x})$</span><span class="n063">, learn a model </span><span class="native-math">${p}_{\theta }\left(\boldsymbol{x}\right)\approx \ {p}_{\mathrm{data} }(\boldsymbol{x})$</span><span class="n063"> such that we can:</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">generate new samples </span><span class="native-math">$\boldsymbol{x}\sim {p}_{\theta }(\boldsymbol{x})$</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">estimate the probability density of the sample </span><span class="native-math">${p}_{\theta }(\boldsymbol{x})$</span></p>
</div>
<div class="native-text n011" data-source-text-id="s049-text4">
<p class="native-paragraph n009"><span class="n008">49</span></p>
</div>
<div id="s049-pic5" class="native-figure n135"><img class="n032" data-src="assets/media/image54.png" alt="DDPMs — original illustration 5"></div>
<div id="s049-pic6" class="native-figure n136"><img class="n032" data-src="assets/media/image55.png" alt="DDPMs — original illustration 6"></div>

---

<!-- .slide: id="source-050" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s050-text2">
<h2 class="native-paragraph n017"><span class="n016">DDPMs</span></h2>
</div>
<div class="native-text n137" data-source-text-id="s050-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Forward diffusion process: Corrupt the image till it becomes Gaussian noise (</span><span class="n120">Fixed process</span><span class="n063">)</span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">Reverse diffusion process: Gradually remove the noise till you recover the image (</span><span class="n120">Learned process</span><span class="n063">)</span></p>
</div>
<div class="native-text n011" data-source-text-id="s050-text4">
<p class="native-paragraph n009"><span class="n008">50</span></p>
</div>
<div id="s050-pic5" class="native-figure n138"><img class="n032" data-src="assets/media/image56.png" alt="DDPMs — original illustration 5"></div>

---

<!-- .slide: id="source-051" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s051-text2">
<h2 class="native-paragraph n017"><span class="n016">DDPMs</span></h2>
</div>
<div class="native-text n139" data-source-text-id="s051-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Both forward and reverse processes are formalized as a Markovian processes, therefore:</span></p>
<p class="native-paragraph n140 native-bullet native-display-math"><span class="native-math">$\displaystyle q({\boldsymbol{x} }_{t}\mid {\boldsymbol{x} }_{t-1},\ \ldots ,\ {\boldsymbol{x} }_{0})=q({\boldsymbol{x} }_{t}\mid \ {\boldsymbol{x} }_{t-1})$</span></p>
<p class="native-paragraph n140 native-bullet native-display-math"><span class="native-math">$\displaystyle p({\boldsymbol{x} }_{t-1}\mid {\boldsymbol{x} }_{t},\ \ldots ,\ {\boldsymbol{x} }_{T})=p({\boldsymbol{x} }_{t-1}\mid {\boldsymbol{x} }_{t})$</span></p>
</div>
<div class="native-text n011" data-source-text-id="s051-text4">
<p class="native-paragraph n009"><span class="n008">51</span></p>
</div>
<div id="s051-pic5" class="native-figure n141"><img class="n032" data-src="assets/media/image58.png" alt="DDPMs — original illustration 5"></div>

---

<!-- .slide: id="source-052" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s052-text2">
<h2 class="native-paragraph n017"><span class="n016">DDPMs</span></h2>
</div>
<div id="s052-pic5" class="native-figure n142"><img class="n032" data-src="assets/media/image59.png" alt="DDPMs — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s052-text4">
<p class="native-paragraph n009"><span class="n008">52</span></p>
</div>

---

<!-- .slide: id="source-053" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div id="s053-pic5" class="native-figure n143"><img class="n032" data-src="assets/media/image60.png" alt="DDPMs — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s053-text4">
<p class="native-paragraph n009"><span class="n008">53</span></p>
</div>
<div class="native-text n015" data-source-text-id="s053-text8">
<h2 class="native-paragraph n017"><span class="n016">DDPMs</span></h2>
</div>

---

<!-- .slide: id="source-054" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s054-text2">
<h2 class="native-paragraph n017"><span class="n016">D3PMs</span></h2>
</div>
<div class="native-text n076" data-source-text-id="s054-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Structured Denoising Diffusion Models in Discrete State-Spaces</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">Traditional models only allow continuous data due to Gaussian Noise which can take on any real value, like pixel intensities in images</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">expands on discrete diffusion models by modelling structured discrete data in the noising process</span></p>
<p class="native-paragraph n106 native-bullet"><span class="n097">discrete-appropriate noise is added: Could be flipping bits (classification), changing tokens (text) or changing between categorical states (categorization)</span></p>
</div>
<div class="native-text n011" data-source-text-id="s054-text4">
<p class="native-paragraph n009"><span class="n008">54</span></p>
</div>

---

<!-- .slide: id="source-055" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s055-text2">
<h2 class="native-paragraph n017"><span class="n016">D3PMs</span></h2>
</div>
<div id="s055-pic5" class="native-figure n144"><img class="n032" data-src="assets/media/image61.png" alt="D3PMs — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s055-text4">
<p class="native-paragraph n009"><span class="n008">55</span></p>
</div>

---

<!-- .slide: id="source-056" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s056-text2">
<h2 class="native-paragraph n017"><span class="n016">D3PMs</span></h2>
</div>
<div class="native-text n145" data-source-text-id="s056-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Quantitative results on text8</span></p>
</div>
<div class="native-text n011" data-source-text-id="s056-text4">
<p class="native-paragraph n009"><span class="n008">56</span></p>
</div>
<div id="s056-pic5" class="native-figure n146"><img class="n032" data-src="assets/media/image62.png" alt="D3PMs — original illustration 5"></div>

---

<!-- .slide: id="source-057" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s057-text2">
<h2 class="native-paragraph n017"><span class="n016">D3PMs</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s057-text4">
<p class="native-paragraph n009"><span class="n008">57</span></p>
</div>
<div class="native-text n147" data-source-text-id="s057-text7">
<p class="native-paragraph n148 native-bullet"><span class="n110">Left: perplexity v.s. sampling iterations for LM1B</span></p>
<p class="native-paragraph n149 native-bullet"><span class="n110">Right: Using a trained D3PM absorbing model for LM1B to (top) generate new sentences and (bottom) reconstruct corrupted examples.</span></p>
</div>
<div id="s057-pic8" class="native-figure n150"><img class="n032" data-src="assets/media/image63.png" alt="D3PMs — original illustration 8"></div>

---

<!-- .slide: id="source-058" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s058-text2">
<h2 class="native-paragraph n017"><span class="n016">D3PMs</span></h2>
</div>
<div class="native-text n151" data-source-text-id="s058-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">Quantitative results on LM1B. Perplexity reported on the test set. Results are reported on two seeds. All models have context window length 128 and 12 layers</span></p>
</div>
<div class="native-text n011" data-source-text-id="s058-text4">
<p class="native-paragraph n009"><span class="n008">58</span></p>
</div>
<div id="s058-pic5" class="native-figure n153"><img class="n152" data-src="assets/media/image64.png" alt="D3PMs — original illustration 5"></div>

---

<!-- .slide: id="source-059" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n011" data-source-text-id="s059-text3">
<p class="native-paragraph n009"><span class="n008">59</span></p>
</div>
<div class="native-text n012" data-source-text-id="s059-text4">
<h2 class="native-paragraph n014"><span class="n013">Diffusion LLMs</span></h2>
</div>

---

<!-- .slide: id="source-060" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s060-text2">
<h2 class="native-paragraph n017"><span class="n016">Mercury</span></h2>
</div>
<div class="native-text n012" data-source-text-id="s060-text3">
<p class="native-paragraph n156"><span class="n154"><a href="https://chat.inceptionlabs.ai/">https://chat.inceptionlabs.ai/</a></span><span class="n155"> </span></p>
</div>
<div class="native-text n011" data-source-text-id="s060-text4">
<p class="native-paragraph n009"><span class="n008">60</span></p>
</div>

---

<!-- .slide: id="source-061" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s061-text2">
<h2 class="native-paragraph n017"><span class="n016">LLaDA</span></h2>
</div>
<div id="s061-pic5" class="native-figure n157"><img class="n032" data-src="assets/media/image65.png" alt="LLaDA — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s061-text4">
<p class="native-paragraph n009"><span class="n008">61</span></p>
</div>

---

<!-- .slide: id="source-062" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s062-text2">
<h2 class="native-paragraph n017"><span class="n016">LLaDA </span><span class="n046">the big picture</span></h2>
</div>
<div class="native-text n158" data-source-text-id="s062-text3">
<p class="native-paragraph n159 native-bullet"><span class="n027">LLaDA is a masked diffusion model that follows standard pretraining and SFT while sampling via diffusion. </span></p>
<p class="native-paragraph n160 native-bullet"><span class="n027">During pretraining, it masks all tokens randomly at ratio </span><span class="native-math">$t\sim U[0,1]$</span><span class="n027"> ; in SFT, only response tokens may be masked. </span></p>
<p class="native-paragraph n160 native-bullet"><span class="n027">The model simulates diffusion from full masking </span><span class="native-math">$(t=1)$</span><span class="n027"> to unmasking </span><span class="native-math">$(t=0)$</span><span class="n027">, predicting all masks simultaneously at each step with flexible remasking.</span></p>
</div>
<div class="native-text n011" data-source-text-id="s062-text4">
<p class="native-paragraph n009"><span class="n008">62</span></p>
</div>
<div id="s062-pic6" class="native-figure n161"><img class="n032" data-src="assets/media/image67.png" alt="LLaDA the big picture — original illustration 6"></div>

---

<!-- .slide: id="source-063" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s063-text2">
<h2 class="native-paragraph n017"><span class="n016">LLaDA </span><span class="n046">competitive performance</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s063-text4">
<p class="native-paragraph n009"><span class="n008">63</span></p>
</div>
<div id="s063-pic10" class="native-figure n162"><img class="n032" data-src="assets/media/image68.png" alt="LLaDA competitive performance — original illustration 10"></div>

---

<!-- .slide: id="source-064" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s064-text2">
<h2 class="native-paragraph n017"><span class="n016">LLaDA </span><span class="n046">competitive performance</span></h2>
</div>
<div id="s064-pic6" class="native-figure n163"><img class="n032" data-src="assets/media/image69.png" alt="LLaDA competitive performance — original illustration 6"></div>
<div class="native-text n011" data-source-text-id="s064-text4">
<p class="native-paragraph n009"><span class="n008">64</span></p>
</div>

---

<!-- .slide: id="source-065" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s065-text2">
<h2 class="native-paragraph n017"><span class="n016">LLaDA </span><span class="n046">competitive performance</span></h2>
</div>
<div class="native-text n164" data-source-text-id="s065-text3">
<p class="native-paragraph n166 native-bullet"><span class="n165">Visualization of the Sampling Process and a Generated Multi-round Dialogue. In the response of LLaDA, darker colors indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.</span></p>
</div>
<div class="native-text n011" data-source-text-id="s065-text4">
<p class="native-paragraph n009"><span class="n008">65</span></p>
</div>
<div id="s065-pic5" class="native-figure n167"><img class="n032" data-src="assets/media/image70.png" alt="LLaDA competitive performance — original illustration 5"></div>

---

<!-- .slide: id="source-066" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s066-text2">
<h2 class="native-paragraph n017"><span class="n016">Dream</span></h2>
</div>
<div class="native-text n011" data-source-text-id="s066-text4">
<p class="native-paragraph n009"><span class="n008">66</span></p>
</div>
<div class="native-text n168" data-source-text-id="s066-text6">
<p class="native-paragraph n098 native-bullet"><span class="n097">Leveraging pre-trained weights from existing autoregressive (AR) models provides a non-trivial initialization strategy for diffusion language models</span></p>
</div>
<div id="s066-pic7" class="native-figure n169"><img class="n032" data-src="assets/media/image71.png" alt="Dream — original illustration 7"></div>
<div aria-hidden="true" class="native-decoration n170"></div>

---

<!-- .slide: id="source-067" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s067-text2">
<h2 class="native-paragraph n017"><span class="n016">Dream</span></h2>
</div>
<div class="native-text n164" data-source-text-id="s067-text3">
<p class="native-paragraph n166 native-bullet"><span class="n165">Comparison of base models on standard evaluation benchmarks. * indicates Dream 7B, LLaDA 8B, Qwen2.5 7B and LLaMA3 8B are evaluated under the same protocol. The best results are bolded and the second best are underlined. </span></p>
</div>
<div class="native-text n011" data-source-text-id="s067-text4">
<p class="native-paragraph n009"><span class="n008">67</span></p>
</div>
<div id="s067-pic5" class="native-figure n171"><img class="n032" data-src="assets/media/image72.png" alt="Dream — original illustration 5"></div>
<div aria-hidden="true" class="native-decoration n172"></div>

---

<!-- .slide: id="source-068" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s068-text2">
<h2 class="native-paragraph n017"><span class="n016">Dream</span></h2>
</div>
<div id="s068-pic5" class="native-figure n173"><img class="n032" data-src="assets/media/image73.png" alt="Dream — original illustration 5"></div>
<div class="native-text n011" data-source-text-id="s068-text4">
<p class="native-paragraph n009"><span class="n008">68</span></p>
</div>

---

<!-- .slide: id="references" class="native-slide" -->

<div aria-hidden="true" class="native-decoration n010"></div>
<div class="native-text n015" data-source-text-id="s069-text2">
<h2 class="native-paragraph n017"><span class="n016">References</span></h2>
</div>
<div class="native-text n076" data-source-text-id="s069-text3">
<p class="native-paragraph n064 native-bullet"><span class="n063">[1] </span><span class="n174"><a href="https://stanford.edu/class/ee367/slides/lecture12.pdf">https://stanford.edu/class/ee367/slides/lecture12.pdf</a></span><span class="n063"> </span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">[2] </span><span class="n174"><a href="https://lemuelpuglisi.github.io/rg2023-ddpm-slides/">https://lemuelpuglisi.github.io/rg2023-ddpm-slides/</a></span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">[3] </span><span class="n174"><a href="https://hkunlp.github.io/blog/2025/dream/">https://hkunlp.github.io/blog/2025/dream/</a></span><span class="n063"> </span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">[4] </span><span class="n174"><a href="https://ml-gsai.github.io/LLaDA-demo/">https://ml-gsai.github.io/LLaDA-demo/</a></span><span class="n063"> </span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">[5] </span><span class="n174"><a href="https://deepmind.google/models/gemini-diffusion/">https://deepmind.google/models/gemini-diffusion/</a></span></p>
<p class="native-paragraph n092 native-bullet"><span class="n063">[6] </span><span class="n174"><a href="https://chat.inceptionlabs.ai/">https://chat.inceptionlabs.ai/</a></span><span class="n063">  </span></p>
</div>
<div class="native-text n011" data-source-text-id="s069-text4">
<p class="native-paragraph n009"><span class="n008">69</span></p>
</div>
