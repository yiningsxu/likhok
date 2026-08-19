# Paper List
[**Notion link**](https://app.notion.com/p/tomotomohsu/3a2d472f23db808880e8cff2c5b5547a?v=3a2d472f23db8001b6d4000c8d067da8&source=copy_link)



[**ClinDet-Bench: Beyond Abstention, Evaluating Judgment Determinability of LLMs in Clinical Decision-Making**](https://arxiv.org/abs/2602.22771)

- **Authors**: Yusuke Watanabe, Yohei Kobashi, Takeshi Kojima, Yusuke Iwasawa, Yasushi Okuno, Yutaka Matsuo 
    
- **Abstract**: Clinical decisions are often required under incomplete information. Clinical experts must identify whether available information is sufficient for judgment, as both premature conclusion and unnecessary abstention can compromise patient safety. To evaluate this capability of large language models (LLMs), we developed ClinDet-Bench, a benchmark based on clinical scoring systems that decomposes incomplete-information scenarios into determinable and undeterminable conditions. Identifying determinability requires considering all hypotheses about missing information, including unlikely ones, and verifying whether the conclusion holds across them. We find that recent LLMs fail to identify determinability under incomplete information, producing both premature judgments and excessive abstention, despite correctly explaining the underlying scoring knowledge and performing well under complete information. These findings suggest that existing benchmarks are insufficient to evaluate the safety of LLMs in clinical settings. ClinDet-Bench provides a framework for evaluating determinability recognition, leading to appropriate abstention, with potential applicability to medicine and other high-stakes domains, and is publicly available.

- **Code**: [https://github.com/yusukewatanabe1208/ClinDet_Benchmark](https://github.com/yusukewatanabe1208/ClinDet_Benchmark)

