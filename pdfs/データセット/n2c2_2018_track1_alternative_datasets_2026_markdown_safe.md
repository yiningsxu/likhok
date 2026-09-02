# n2c2 2018 Track 1 の代替候補となる EHR / Clinical Reasoning データセット調査

**調査時点:** 2026年9月2日

> **表示互換性について**  
> この版では、Markdown環境によって表示が崩れやすい高度なLaTeX構文を避け、重要な構成は通常のMarkdown、表、インラインコードで記述しています。

## 結論

2026年9月2日時点では、**n2c2 2018 Track 1をそのまま置き換えられる単一の公開・申請可能データセットは確認できない**。

n2c2 2018 Track 1の特徴は、

**n2c2 2018 Track 1 の基本構造**

`複数時点の実臨床記録 × 固定された13個の適格基準 → {met, not met}`

という、**実患者の縦断的 free-text EHR に対する patient–criterion 単位の専門家ラベル**を持つ点にある。

現在利用可能な候補は、大きく以下のように整理できる。

- **データ構造が最も近い**：MIMIC-III-Ext-VeriFact-BHC
- **臨床試験適格性というタスクが最も近い**：TREC Clinical Trials＋TrialMatchAI／NLI4PR
- **実患者の縦断EHR推論として最も有力**：ER-Reason
- **広範な縦断カルテ読解として最も有力**：MedAlign
- **n2c2型データを新しく構築する基盤**：MIMIC-IV＋MIMIC-IV-Note

---

## 候補の総合比較

| 候補 | n2c2に近い部分 | 2026年の新規取得 | 主な相違点 |
|---|---|---|---|
| **VeriFact-BHC** | 実患者、複数EHR文書、criterion-likeな命題、専門家3値ラベル、根拠検索 | **申請可能**。PhysioNet credentialing＋CITI＋DUA | clinical-trial criteriaではなく、退院要約中の命題の事実検証 |
| **TREC CT＋TrialMatchAI** | 患者情報と実際のtrial eligibility criteriaの照合、criterion-level reasoning | **公開取得可能** | 患者記録が主として合成・短文で、実縦断EHRではない |
| **ER-Reason** | 実患者、複数受診・複数ノート、診断・治療・disposition推論 | **申請可能**。PhysioNet＋研究計画ごとの著者審査 | 適格基準分類ではない |
| **MedAlign** | 実患者の長大な縦断EHR、timeline理解、multi-document synthesis | **申請可能**。Redivisで審査 | open-ended instruction followingであり、固定criterion分類ではない |
| **MIMIC-IV＋Note** | 実EHR、自由記述、構造化データ、時間情報 | **申請可能** | eligibility gold labelを自分で作る必要がある |
| **MedicalBench** | note–concept verification、暗黙的臨床推論、根拠文ラベル | **申請可能** | 405例、単一退院要約中心で縦断性が弱い |
| **SBDH** | alcohol/drug useなど一部のn2c2 criterionに直接対応 | **申請可能** | criterion範囲が社会・行動因子に限定される |
| **MedAgentBench** | EHR上でのagentの検索・操作・計画を評価可能 | **公開取得可能** | 自由記述カルテではなくFHIR中心、trial matchingではない |

---

# 1. 単一データセットとして最も近い：VeriFact-BHC

## データ構造

**MIMIC-III-Ext-VeriFact-BHC v1.1.0** は、100患者について human-written または LLM-written の Brief Hospital Course を命題に分解し、合計13,070命題を収録したデータセットである。

各命題について、その患者の残りの臨床ノートを参照し、

`y ∈ {Supported, Not Supported, Not Addressed}`

を3名の臨床家が付与し、多数決・追加評価・adjudicationによって最終ラベルが作られている。個々の評価者のラベルも保持される。

患者ごとの参照EHRには、対象となる最終退院要約を除く臨床ノートが含まれる。

n2c2との対応は、おおよそ

**n2c2との概念的対応**

| n2c2 2018 Track 1 | VeriFact-BHC |
|---|---|
| eligibility criterion | proposition |
| longitudinal patient record | patient-specific reference EHR |
| met | Supported |
| not met | Not Supported |

とみなせる。

さらに `Not Addressed` が独立しているため、

```text
反証された
vs.
記録中に情報がない
```

を区別できる。

### 研究上の利点

特に次の multi-agent 設計に向いている。

- 各agentが異なるノートや時点を検索する
- 複数agentが同じ命題を独立に判定する
- evidence retrieval agent と verifier agent を分離する
- disagreementを adjudicator agent が解消する
- `Not Addressed` を利用してabstentionを評価する

個々の臨床家ラベルと最終adjudicated labelがあるため、**人間集団の不一致とLLM agent集団の不一致を比較する研究**にも利用可能。

### 取得状況

PhysioNet上で新規アクセス経路が存在し、通常は以下が必要。

1. PhysioNet credentialed user になる
2. CITI の「Data or Specimens Only Research」を修了する
3. データセットのDUAに同意する

PhysioNetでは新規credentialingに遅延が生じる場合があるため、申請期間は余裕を持つべき。

**総合評価：n2c2の「縦断カルテから固定された記述の真偽を判断する」という構造に最も近い単一データセット。**

---

# 2. 臨床試験適格性という意味で最も近い：TREC＋TrialMatchAI＋NLI4PR

## TREC Clinical Trials 2021/2022

TREC Clinical Trialsでは、

```text
patient description
→
eligible clinical trialsのランキング
```

を扱う。

2022トラックでは、医学的訓練を受けた作成者による5–10文程度の合成患者記述と、ClinicalTrials.govのtrial corpusを使用し、trialごとに

`{eligible, excludes, not relevant}`

のjudgmentを持つ。

n2c2と異なり、患者側は実EHRではなく**合成された入院記録風の短い記述**である。また基本的なgold labelはpatient–criterion単位ではなくpatient–trial単位。

それでも、

- inclusion criteria
- exclusion criteria
- 時間条件
- 薬剤・診断・検査値
- 情報不足と明示的除外の区別

を扱うため、**臨床試験適格性という問題設定自体は非常に近い**。

## TrialMatchAI

TrialMatchAIは、患者情報を各trialのinclusion/exclusion criterionと個別に照合する。

criterion-level判定の例：

`{Met, Not Met, Unclear, Irrelevant}`

n2c2より細かいラベル構造を持つ。

必要であれば二値化できるが、

`not met ≠ unclear`

なので、研究上は多値のまま扱う方が望ましい。

公開部分には、合成患者、expert-examined patient–criterion pairs、parsed clinical-trial databaseなどが含まれる。一方、実患者コホートの一部は個別申請が必要。

## NLI4PR

NLI4PRは、患者profileとclinical trialの適格条件を照合するNatural Language Inferenceデータセット。

概念的には、

```text
patient profile
+
eligibility statement
→
entailment / contradiction
```

というn2c2に近い定式化が可能。

### この系列の位置づけ

申請待ちなしで clinical-trial eligibility の実験を始める場合には、

**推奨構成:** `TREC 2021/2022 + TrialMatchAI + NLI4PR`

が有力。

multi-agent研究では、

1. trial retrieval agent
2. inclusion-criterion agent群
3. exclusion-criterion agent群
4. temporal reasoning agent
5. evidence verifier
6. aggregation/adjudication agent

という分解が自然。

ただし、患者記録の現実性・長さ・縦断性ではn2c2より簡単。

---

# 3. 実患者の縦断的clinical reasoning：ER-Reason

ER-Reasonは、実EHRから構成されたclinical reasoningデータセット。

主な内容：

- 数千患者規模
- 複数の救急受診
- 多数の臨床ノート
- discharge summary
- progress note
- H&P
- consult
- imaging
- ER provider note
- triage
- treatment
- disposition
- final diagnosis
- 一部に医師作成のreasoning rationale

n2c2との共通点は、

```text
longitudinal notes
→
clinical decision
```

という点。

一方、出力は trial criterion の met/not-met ではなく、

- 鑑別診断
- rule-out reasoning
- 治療選択
- 入院・帰宅等のdisposition
- 最終診断

である。

### 取得状況

通常のPhysioNet credentialing、CITI、DUAに加え、研究目的についてdataset contributorの追加審査が必要になる。

### 研究上の位置づけ

**診断を行うLLM agent、複数agentによる鑑別診断、逐次的意思決定を研究する場合には、n2c2よりER-Reasonの方が研究目的に直接合う可能性がある。**

---

# 4. 長大な縦断カルテ読解：MedAlign

MedAlignは、

- 実患者
- 数万件規模の臨床ノート
- 多数のnote type
- OMOP clinical events
- clinician-curated instructions
- clinician-written reference responses

からなる、縦断EHRの instruction-following benchmark。

n2c2と共通するのは、モデルが単一の短い症例文ではなく、**大量の患者別文書から必要な情報を探し、時間関係を統合して答える**点。

ただし、MedAlignの課題は固定されたcriterion分類ではなく、臨床家が作成した自然言語instructionへのopen-ended responseである。

したがって、

- factual correctness
- evidence grounding
- completeness
- temporal consistency
- clinician preference

などの評価が重要。

### 取得状況

Redivis経由で申請可能。

一般に必要となるのは、

1. 大学・政府・企業研究機関のメールアドレス
2. HIPAA対応CITI training
3. 研究用途の説明
4. MedAlign DUAへの同意
5. 暗号化・アクセス制御された保存環境

など。

またMedAlignは基本的に**test-only benchmark**であり、学習・fine-tuningには制約がある。

### 研究上の位置づけ

- long-context chart review
- multi-agent synthesis
- timeline reasoning
- error analysis

に特に有力。

---

# 5. n2c2型データを新しく構築するなら：MIMIC-IV＋MIMIC-IV-Note

MIMIC-IVは病院・救急・ICUの構造化EHRを持つ大規模データセットで、MIMIC-IV-Noteの自由記述データと患者ID・入院IDで結合できる。

新規利用では通常、

- PhysioNet credentialing
- CITI training
- DUA

が必要。

MIMIC-IV自体にはclinical-trial eligibility gold labelは存在しない。

そのため、n2c2に最も近い新しいベンチマークを作るなら、

`D = {(R_i, c_j, y_ij, E_ij, r_ij)}`

とし、

- `R_i`：患者 `i` の時系列EHR
- `c_j`：ClinicalTrials.gov等から取得した適格基準
- `y_{ij}∈{met,not met,insufficient}`
- `E_{ij}`：根拠となる文・検査値・時点
- `r_{ij}`：臨床家による短い判断理由

を保存する設計が考えられる。

特に、

`not met ≠ insufficient evidence`

を区別すると、hallucinationや過剰推論を独立に評価できる。

### 推奨される設計上の注意

- データ分割は文書単位ではなくpatient単位
- 同一患者の複数入院をtrain/testにまたがせない
- trial registry snapshotを固定する
- criterion-level gold labelを保持する
- evidence spanを保存する
- temporal relationを保存する
- annotatorごとの個別判断も可能なら保持する
- adjudicated goldを別途作成する

---

# 6. 補助的に有用なデータセット

## MIMIC-IV-Ext-MedicalBench

MedicalBenchは、退院要約と候補clinical conceptについて、

```text
(note,concept)
→
(supported?,evidence spans,rationale)
```

を評価する小規模・高品質データセット。

- 明示的な診断
- 検査所見や治療から暗黙的に推論される概念
- semantic hard negatives
- evidence span
- rationale

などを扱える。

縦断性は弱いが、**criterion-level verifierの精度と根拠忠実性を評価する補助セット**として有用。

---

## MIMIC-III-Ext-SBDH

Social and Behavioral Determinants of Healthを対象とし、

- employment
- housing
- marital status
- alcohol use
- tobacco use
- drug use

などをannotationする。

n2c2の

- ALCOHOL-ABUSE
- DRUG-ABUSE

に近いcriterionを含むため、社会・行動因子に限定した外部検証に利用できる。

---

## ArchEHR-QA

患者からの質問に対し、MIMIC由来のclinical note excerptsから根拠文を引用して回答するデータセット。

適格性分類ではないが、

- evidence retrieval
- grounded answering
- clinical QA

の評価に利用可能。

---

## MedAgentBench

MedAgentBenchは、FHIRベースのEHR環境上でagentが、

- EHR検索
- データ取得
- medication確認
- lab確認
- diagnosis確認
- 複数API callの計画
- FHIR操作

などを実行するinteractive benchmark。

自由記述カルテやtrial matchingそのものではないが、**multi-agent systemのtool use、役割分担、検索計画、行動選択**の評価に有用。

---

# 推奨する研究構成

## 1. すぐ実験を始める場合

申請待ちなしで進めるなら、

**推奨構成:** `TREC 2021/2022 + TrialMatchAI + NLI4PR + MedAgentBench`

が有力。

- TREC／TrialMatchAI：patient–criterion reasoningとaggregation
- NLI4PR：eligibility NLI
- MedAgentBench：agent検索・tool-use能力

を評価できる。

ただし、**実縦断EHRでの性能をこの構成だけから主張することはできない**。

---

## 2. 実EHRを含む現実的な構成

**推奨構成**

- `VeriFact-BHC` — 命題単位のEHR根拠検証
- `ER-Reason` — 縦断的診断・治療推論
- `MedAlign` — 長大なカルテの統合評価
- `TREC / TrialMatchAI` — clinical-trial semantics

という複合ベンチマークが強い。

役割は以下。

- **TrialMatchAI/TREC**：適格基準ごとのagent分解と集約
- **VeriFact-BHC**：実EHRに対するevidence retrievalとverdict
- **ER-Reason**：診断・治療・dispositionの逐次推論
- **MedAlign**：長文脈・複数文書統合の最終評価

---

## 3. n2c2の直接後継を作る場合

研究の中心が、

> 複数agentを用いて、長い患者記録から複数の適格基準を判定すると、単一agentよりどの条件で優れるか

であるなら、最も適切なのは、

**推奨方針:** `MIMIC-IV / MIMIC-IV-Note 上に小規模な patient–criterion gold set を新規構築する`

という方針。

少なくとも以下を保持することが望ましい。

- criterionごとの独立ラベル
- 情報不足ラベル
- 根拠文
- 根拠時点
- temporal relation
- 複数annotatorの個別判断
- adjudicated gold
- patient-level split

これにより、

- agent数
- agent能力差
- agent専門性
- 情報分割
- communication protocol
- 誤り相関
- aggregation rule

などを理論モデルの変数に対応させやすくなる。

---

# データ利用時の重要な制約

PhysioNet/MIMIC由来データについては、データ使用契約上、第三者の外部APIへの送信が問題になる場合がある。

したがって、

- VeriFact-BHC
- MedicalBench
- ArchEHR-QA
- MIMIC-IV/Note

などは、**ローカルまたは所属機関内の管理された計算環境で実行する設計が基本**。

MedAlignについても、非HIPAA-compliantなcommercial APIへの送信には制約がある。

実験開始順としては、

1. 公開TREC／TrialMatchAI／NLI4PRでアルゴリズム開発
2. 並行してMedAlign、VeriFact-BHC、ER-Reasonへ申請
3. 承認後、ローカルモデルで実EHR評価
4. 必要ならMIMIC-IV上でn2c2型annotationを追加

という流れが現実的。

---

# 総合評価

単一代替としては、

**単一代替として最有力:** `VeriFact-BHC`

が最もn2c2 2018 Track 1のデータ構造に近い。

臨床試験適格性というタスクそのものを重視するなら、

**臨床試験適格性タスクとして最有力:** `TREC Clinical Trials + TrialMatchAI`

が有力。

実縦断EHRでclinical reasoningを評価するなら、

**実縦断EHRのclinical reasoningでは有力:** `ER-Reason`

が重要。

長大な縦断カルテの統合推論・multi-document reasoningを測るなら、

**長大な縦断カルテの統合推論では有力:** `MedAlign`

が有力。

最終的にn2c2型のmulti-agent研究を厳密に行うなら、

**最終的な推奨構成:** `既存ベンチマーク群 + MIMIC-IV 上に構築した新規 patient–criterion gold set`

という構成が最も研究上の自由度と妥当性が高い。
