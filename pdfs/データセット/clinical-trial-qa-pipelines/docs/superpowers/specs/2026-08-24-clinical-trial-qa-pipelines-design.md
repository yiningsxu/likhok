# Clinical Trial QA Pipelines Design

## 1. 目的

同一の診療記録と23個の治験基準質問に対し、6種類のLLMパイプライン（P1〜P6）を同じ入出力契約で実行し、回答精度、根拠忠実性、数値完全性、棄却性能、費用・遅延を比較できる研究用コードを構築する。

本実装は研究評価用であり、臨床判断や被験者登録を自動確定する医療機器ではない。原文引用が検証できない出力は採用しない。

## 2. 入力データ

既存CSVの必須列は次の8列とする。

| 列 | 意味 |
|---|---|
| `text` | 診療記録全文 |
| `note_id` | 記録識別子 |
| `hadm_id` | 入院識別子 |
| `criterion` | 23基準の識別子 |
| `question_type` | `yes` または `numeric` |
| `question` | 質問文 |
| `answer` | 既存の専門家正解 |
| `not_specified` | 既存データにおける未記載フラグ |

ローダは1記録23行を1つの `NoteCase` に集約し、同一 `note_id` 内で `text` と `hadm_id` が一致すること、および質問が重複しないことを検証する。元CSVは変更しない。

## 3. 質問仕様

23問はコード化した `QuestionSpec` で管理する。15問は記載判定、8問は数値抽出である。

- 記載判定の文書状態: `yes`, `no`, `not_documented`, `indeterminate`
- 数値判定の文書状態: `value_available`, `not_documented`, `indeterminate`
- 数値の集約: `chads2`, `blood_glucose`, `CREAT`, `AST`, `BILI` は最大値、`lvef`, `PLT`, `HGB` は最小値
- `lvef` の既存データセット互換値は、最小値が55%以上なら55へクリップする。抽出候補自体は原値を保持する。

各仕様は別名、想定単位、ルーティングラベル、時間条件、対象者条件を持つ。数値比較はLLMに任せず、検証済み候補からPythonで決定論的に計算する。

## 4. 共通出力契約

1質問につき、最終出力を `QuestionResult` とする。

```json
{
  "note_id": "string",
  "criterion": "PLT",
  "question_type": "numeric",
  "document_status": "value_available",
  "answer": 91.0,
  "unit": "K/uL",
  "evidence": [
    {
      "quote": "PLT 91 K/uL",
      "start_char": 120,
      "end_char": 132,
      "section_id": "labs-2",
      "source_scope": "routed",
      "raw_value": "91",
      "normalized_value": 91.0,
      "unit": "K/uL",
      "time_text": null
    }
  ],
  "candidate_values": [
    {
      "quote": "PLT 91 K/uL",
      "start_char": 120,
      "end_char": 132,
      "section_id": "labs-2",
      "source_scope": "routed",
      "raw_value": "91",
      "normalized_value": 91.0,
      "unit": "K/uL",
      "time_text": null
    }
  ],
  "inference": null,
  "confidence": 0.91,
  "provenance": [],
  "validation_errors": []
}
```

規則:

1. `quote` は診療記録中の完全一致部分文字列でなければならない。
2. `start_char:end_char` は `quote` と一致するようコード側で再計算する。
3. 数値質問では、対象検査の全候補値と各引用を `candidate_values` に保持する。
4. 数値の最終値は検証済み全候補から決定論的reducerが算出する。
5. 考察は `inference` にのみ記録し、原文引用と混ぜない。
6. verifier/aggregatorは根拠台帳にない引用をそのまま採用できない。
7. 根拠が不十分なら過度な確信を避け、`not_documented` または `indeterminate` を選ぶ。

## 5. 共通コンポーネント

### 5.1 LLM adapter

`LLMClient.generate(request) -> LLMResponse` を唯一の外部モデル境界とする。

- `OpenAICompatibleClient`: `/chat/completions` 互換HTTP APIを環境変数で呼ぶ。
- `ScriptedLLMClient`: テストとdry-run用。実APIやAPIキーを必要としない。
- 各呼び出しはモデル名、温度、入力/出力文字数、遅延、エラー、run IDを記録する。

### 5.2 構造化出力

LLMにはJSONのみを要求する。コード側は構文、enum、型、必須フィールドを検証する。壊れたJSONは上限付きの再試行対象とし、上限後はエラー結果として記録する。

### 5.3 根拠検証

`EvidenceValidator` は引用を全文から完全一致検索し、複数一致時はLLMのoffsetを候補選択にのみ使う。完全一致しない引用は無効化する。最終回答に残るすべての引用は検証済みである。

### 5.4 数値処理

`NumericReducer` は次を行う。

1. LLMが返した候補の引用を検証する。
2. 引用内に候補のraw valueが存在することを確認する。
3. 桁区切り、不等号、百分率、単位を保存しつつ数値を正規化する。
4. 質問仕様どおりに最小値または最大値を計算する。
5. 数値のない検査言及は `indeterminate`、言及自体がなければ `not_documented` とする。

単位変換は暗黙に行わない。比較不能な単位が混在した場合は `indeterminate` とし、候補は保持する。

### 5.5 セクション分割とルーティング

セクション分割は見出しと空行を使う決定論的処理を基本とする。セクションおよび質問には複数ラベルを許可する。

- ラベル: `diagnosis`, `history`, `medication`, `procedure`, `laboratory`, `cardiology`, `neurology`, `psychiatry`, `bleeding`, `capacity`, `other`
- ルータは再現率優先でtop-kを返す。
- 該当候補がない場合も全文fallbackを保持する。
- P4〜P6では、最後に全文verifierを必ず実行してルーティング漏れを監査する。

### 5.6 集約と反論

- 複数モデルの一次回答は互いに見せず独立生成する。
- aggregatorは回答、検証済み根拠、候補数値、信頼度を比較する。
- 同票や根拠衝突は `indeterminate` を優先する。
- role debateは最大2ラウンド。無限反論は行わない。
- 役割は疾患分野よりエラー機序で分ける: assertion/negation/experiencer、temporality、numeric completeness、evidence fidelity、adjudication。

## 6. 六つのパイプライン

| ID | 入力 | 一次推論 | 集約 | 最終監査 |
|---|---|---|---|---|
| P1 | 全文 | 単一LLM、1問ずつ | なし | 共通postprocessor |
| P2 | 全文 | 複数独立LLM、1問ずつ | evidence-gated aggregator | 共通postprocessor |
| P3 | 全文 | 同一LLMの複数エラー役割 | bounded adjudication | 共通postprocessor |
| P4 | routed sections | 単一LLM | なし | 全文verifier |
| P5 | routed sections | 複数独立LLM | evidence-gated aggregator | 全文verifier |
| P6 | routed sections | 複数エラー役割 | bounded adjudication | 全文verifier |

P1〜P6は共通の `Pipeline.run_case(case) -> list[QuestionResult]` を実装する。質問単位の失敗は他の質問を中断せず、エラーを結果に記録する。

## 7. 評価

### 7.1 データ分割

- 分割単位は `note_id`。23行を跨いだrow-level splitは禁止する。
- seed付きpatient/note-level holdoutを提供する。
- promptやfew-shot例を評価記録から作らない。

### 7.2 主要指標

- 既存ラベル互換 accuracy（boolean Yes/No、numeric exact/tolerance）
- criterion macro accuracy
- boolean macro-F1 と balanced accuracy
- 数値 MAE、許容誤差内率、missingness accuracy
- selective accuracy / coverage（棄却を含む）
- evidence exact-match validity
- numeric candidate completeness（根拠アノテーション追加後）
- latency、LLM calls、入力/出力文字数

既存CSVには根拠アノテーションがないため、根拠妥当性は「引用が原文に存在するか」までを自動評価し、根拠再現率は将来のgold span列がある場合のみ算出する。

### 7.3 主要比較

P1〜P6を同一分割、同一質問順、同一temperature方針で比較する。モデル多様性の効果とサンプリング多様性の効果を混同しないため、各runのprovider/model/prompt hashを保存する。

## 8. CLIと設定

```bash
clinical-trial-qa validate-data --input annotated_apixaban_combined.csv
clinical-trial-qa run --config configs/example.json --input annotated_apixaban_combined.csv --pipeline p5 --limit-notes 1
clinical-trial-qa evaluate --gold annotated_apixaban_combined.csv --predictions outputs/p5.jsonl
clinical-trial-qa split --input annotated_apixaban_combined.csv --output-dir splits --seed 42
```

YAMLではなく標準JSON設定を採用し、追加依存を避ける。API keyは設定ファイルへ書かず、指定した環境変数名から読む。

## 9. セキュリティと再現性

- 診療記録全文を通常ログへ出力しない。
- 配布アーカイブに入力CSV、予測結果、API key、cacheを含めない。
- 外部APIへ送る前に利用者がデータ利用契約と匿名化方針を確認する必要がある旨をREADMEに明記する。
- run manifestに設定、コードversion、seed、prompt version、モデル名を保存する。
- temperature 0を既定とし、self-consistencyを行う場合だけ明示的に変更する。

## 10. テスト戦略

1. CSV集約と不整合検出
2. `note_id` 単位の分割と漏洩防止
3. 引用完全一致、offset再計算、存在しない引用の拒否
4. 数値候補の最小/最大、LVEFクリップ、混在単位、候補なし
5. 4値/3値状態の正規化
6. ルーティングの再現率fallback
7. P1〜P6の呼び出し構成をscripted clientで検証
8. aggregatorの同票棄却と新規根拠拒否
9. verifierのaccept/revise/abstain
10. CLIのvalidate/run/evaluate dry-run
11. 添付CSVに対するread-only schema validation

## 11. 非目標

- 特定LLMベンダーへの固定
- 診療記録の自動匿名化
- 臨床登録判断の自動実行
- 既存CSVの専門家ラベルの変更
- 全単位体系の汎用変換
- ベクトルDBや分散実行基盤
