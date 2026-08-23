# HCMAIC OCR channel — trạng thái và pipeline hiện tại

> Cập nhật: 2026-08-18  
> Execution status: **ENGINEERING_ARTIFACT_COMPLETE cho canary 120 frame**  
> Retrieval quality: **UNVALIDATED** — chưa có OCR-specific qrels/benchmark được duyệt.  
> Phạm vi: OCR là channel độc lập; mọi join downstream dùng `frame_uid`/`crop_uid`.

## 0. Kết luận hiện tại

OCR đã có execution canary, không còn chỉ là design. Ba recognizer đã chạy trên cùng detector và cùng 1.127 crop:

| Artifact | Model | Coverage | Failure | Đánh giá |
|---|---|---:|---:|---|
| `hcmaic-ocr-vietocr-canary-20260818` | `VietOCR:vgg_transformer`, `vietocr==0.3.13` | 1.127/1.127 | 0 | ENGINEERING_PROXY |
| `hcmaic-ocr-en-v5-rec-canary-20260818` | `en_PP-OCRv5_mobile_rec` | 1.127/1.127 | 0 | ENGINEERING_PROXY, benchmark |
| `hcmaic-ocr-v6-medium-rec-canary-20260818` | `PP-OCRv6_medium_rec` | 1.127/1.127 | 0 | ENGINEERING_PROXY |

Ba output cùng selection hash:

```text
4cb2318bf89a744cbba4ef7d011cd17c207f360faef73433b6224ed84c9ce2bc
```

Detector/crop dùng chung là `PP-OCRv6_medium_det`. Canary gồm 120 frame, trong đó 116 frame có line detection; 4 frame không có line là `NO_TEXT`, không phải inference failure.

Review tri-model local:

```text
tmp/ocr-tri-review-bundle-20260818
http://127.0.0.1:8766/
```

Ví dụ `VF RIB 112`: PP6 đọc `VF RIB 112`, PP5 đọc `VFRIB 112`, VietOCR đọc sai `Wribitz`. Đây là bằng chứng pilot ủng hộ PP6, chưa phải quality proof toàn corpus.

## 1. Quyết định stack

Pipeline production đề xuất:

```text
keyframe + metadata
  -> PP-OCRv6_medium_det
  -> padded/rectified line crop
  -> PP-OCRv6_medium_rec + VietOCR
  -> raw/normalized/candidate fields
  -> line + temporal-span aggregation
  -> BM25/Elasticsearch OCR index
  -> optional BGE-M3 text index
  -> OCR results fused with ASR/visual/object
```

Quyết định model:

1. `PP-OCRv6_medium_det`: detector dùng chung, đã được dùng trong canary.
2. `PP-OCRv6_medium_rec`: recognizer chính cho tiếng Anh, số, mã, formula-like text và text tổng quát.
3. `VietOCR:vgg_transformer`: recognizer bổ sung cho tiếng Việt có dấu. Chạy trên cùng crop để giữ recall; không overwrite PP6.
4. `PP-OCRv5`: không chạy full corpus. Chỉ giữ artifact canary để benchmark/debug.
5. PaddleOCR-VL hoặc OCR-VLM: chưa đưa vào P0; chỉ xem xét selective fallback cho slide/bảng/công thức khó nếu qrels chứng minh lợi ích.

Không chọn một model duy nhất ở bước extraction. Nếu PP6 và VietOCR khác nhau, giữ cả hai candidate để retrieval quyết định theo field/model và provenance.

## 2. Identity và bất biến

- Keyframe v1 và input selection là immutable; OCR tạo artifact version riêng.
- Stable frame identity: `frame_uid=video_id:source_frame_idx`.
- Stable OCR identity: `crop_uid`; line-level key là `crop_uid` + `frame_uid`.
- `faiss_row`, parquet row, Elasticsearch `_id` và local row number chỉ là index-local, không phải identity.
- Mọi OCR row/span phải map ngược được tới `video_id`, `shot_id`, `timestamp_ms`, `frame_uid`, bbox/polygon và crop.
- Giữ raw text và raw detector/recognizer response; normalize/correction chỉ là field phụ.
- Không coi Kaggle `COMPLETE`, đủ row, hash vector hay index build thành retrieval quality.
- Mỗi shard/phase phải có manifest, failure ledger, identity hash và model/runtime provenance.

## 3. Extraction pipeline

### 3.1 Input và preflight

Đọc keyframe JPEG cùng metadata đã version. Trước inference:

- kiểm tra image root và first/middle/last sample;
- kiểm tra selection hash và frame coverage;
- ghi `input_manifest`, model/revision, runtime, device và batch config;
- failure ledger phải phân biệt `NO_TEXT`, `READ_FAILED`, `INFERENCE_FAILED`, `PARSE_ERROR`.

### 3.2 Detection

Chạy `PP-OCRv6_medium_det` một lần trên mỗi frame. Với mỗi polygon/box, ghi:

```text
crop_uid, frame_uid, video_id, shot_id, source_frame_idx, timestamp_ms
bbox, polygon, det_score, detector_model, detector_revision
```

Không bỏ box nhỏ chỉ vì diện tích; text nhỏ có thể là tên người, địa điểm, mã hoặc biển hiệu.

### 3.3 Crop

- Expand khoảng 5–10% để tránh mất dấu ở mép.
- Perspective rectify cho line nghiêng.
- Giữ aspect ratio; không ép line dài thành ảnh vuông.
- Upscale/sharpen chỉ tạo variant phụ và ghi rõ `preprocess_variant`.
- Line quá dài chỉ được chia cửa sổ overlap khi cần; vẫn giữ mapping về crop gốc.

### 3.4 Recognition

Chạy PP6 và VietOCR trên cùng crop trong phase riêng/resumable. Mỗi output phải có:

```text
ocr_text_raw, ocr_text_norm, rec_score
recognizer_model, recognizer_revision, confidence_status
crop_uid, frame_uid, bbox, polygon
```

Quy tắc candidate:

- giống nhau sau normalize: đánh dấu `agreement=true`;
- khác nhau: lưu cả `pp6_text` và `vietocr_text`, không tự overwrite;
- rỗng hoặc crop quá khó: `LOW_CONF/EMPTY`, không hallucinate;
- PP5 chỉ tham gia so sánh offline, không phải production fallback.

Confidence giữa các model không được so sánh như cùng một xác suất. Threshold selection/fallback phải tune bằng qrels hoặc tập OCR gán nhãn nhỏ.

### 3.5 Charset gate

Trước full run phải kiểm tra ít nhất:

```text
ĐỘI · TRƯỜNG · NGUYỄN · CHÍNH · THÀNH PHỐ · CỘNG HÒA
VF RIB 112 · 18:34:33 · 2Fe3+ + Fe → 3Fe2+
```

Không suy ra charset từ tên model. Phải lưu revision/dictionary thực tế và kiểm tra dấu tiếng Việt, số, ký hiệu, công thức.

## 4. Normalize và temporal aggregation

Lưu song song:

```text
text_raw       # output nguyên bản
text_nfc       # Unicode NFC
text_lower     # lowercase nhưng giữ dấu
text_folded    # lowercase + bỏ dấu, field phụ
text_char      # character n-gram/triage field
```

Không xóa số, `%`, `+`, `/`, `-`, dấu thập phân hoặc ký hiệu công thức. Correction/alias chỉ là sidecar; không thay raw.

Các line lặp trong cùng shot được group bằng shot/time, vị trí box, text similarity và chất lượng ảnh:

```text
span_uid, start_s, end_s, canonical_text
support_count, representative_frame_uid, member_frame_uids
raw_observations
```

Consensus chỉ tạo `canonical_text`; vẫn giữ per-frame observations. Không dedup global toàn video, vì cùng text có thể xuất hiện lại ở thời điểm khác.

## 5. OCR index

### 5.1 Lexical index — P0

Index line-level để giữ bbox và localization; có thể tạo index phụ ở frame/span-level để ranking:

```text
frame_uid, video_id, shot_id, timestamp_s, line_index
bbox/polygon, text_raw, text_nfc, text_folded, text_char
pp6_text, vietocr_text, det_score, rec_score
span_uid, support_count, repeat_ratio, source_model_revision
```

Truy vấn nên có các branch:

```text
exact/phrase > text_nfc > text_folded > char 3–5 gram > fuzzy có guard
```

Fuzzy không dùng mạnh cho token ngắn, năm, số, mã hoặc công thức. Character n-gram giúp chịu lỗi như `VFRIB112`/`VF RIB 112` hoặc mất dấu.

### 5.2 Text dense — P1/P2

Nếu thêm BGE-M3:

```text
model = BAAI/bge-m3
dimension = 1024
similarity = cosine
```

Embed chủ yếu ở frame text hoặc temporal span text, không biến mỗi line ngắn thành hàng triệu vector. BGE là channel bổ sung; BM25 vẫn giữ exact entity và số.

Reranker chỉ chạy top candidate, không rerank toàn corpus. Mọi vector/index phải lưu revision, dimension, normalization, source manifest hash và document counts.

## 6. Retrieval contract

OCR trả về line/frame/shot evidence, sau đó mới fusion với ASR, visual và object:

```text
OCR BM25
OCR BGE (optional)
ASR BM25/BGE
SigLIP2/Qwen visual
object detection (optional)
  -> rank fusion/RRF
  -> collapse theo frame
  -> temporal diversity theo shot/video
```

Fusion dùng rank/normalized scores, không cộng raw score khác thang. Query original luôn phải chạy; query expansion chỉ là branch phụ khi retrieval core đã ổn định.

Kết quả cuối phải giữ `matched_field`, `matched_model`, `crop_uid`, `frame_uid`, bbox, timestamp và score provenance.

## 7. Artifact/manifest bắt buộc

`ocr_manifest.json` tối thiểu phải ghi:

```text
status, quality_status
input dataset/version và input manifest hash
selection/frame/shot/crop counts
frame_uid identity hash
detector + recognizer model/revision/dictionary hash
preprocess parameters
runtime/device/batch config
output row counts
failure_ledger path/hash
raw JSONL/Parquet path/hash
```

Index manifest phải ghi mapping/analyzer hash, n-gram range, BGE model/dimension/normalization, document counts, unique/missing identity counts và source OCR manifest hash.

## 8. Phase triển khai hiện tại

### Đã hoàn thành — ENGINEERING_PROXY

- Detector/crop canary: 120 frame, 1.127 crop, shared selection.
- VietOCR recognition: 1.127/1.127, failure ledger rỗng.
- PP-OCRv5 recognition: 1.127/1.127, failure ledger rỗng; chỉ benchmark.
- PP-OCRv6 recognition: 1.127/1.127, failure ledger rỗng.
- Tri-model local viewer đã join và kiểm tra HTTP/API thành công.

### P0 tiếp theo — OCR full extraction

1. Chạy detector/crop canonical trên full keyframe corpus hoặc dùng bundle detector đã được audit.
2. Chạy PP6 + VietOCR trên cùng crop, resumable theo shard/phase.
3. Ghi raw/normalized/candidate fields và failure ledger.
4. Preflight count/hash/schema trước khi upload/merge.
5. Review stratified sample; không gọi quality COMPLETE.

### P1 — OCR lexical retrieval

- Build line/frame/span documents.
- Thêm NFC, folded, character n-gram và phrase branch.
- Test exact entity, no-accent, typo, số, mã, formula và ticker.

### P2 — Dense/rerank và fusion

- BGE-M3 ở frame/span text.
- RRF với ASR/visual/object.
- Rerank top 50–100 nếu latency cho phép.
- Chỉ tune weights sau qrels.

## 9. Quality gate

Execution gate cần kiểm tra:

- input/image preflight GREEN;
- selection hash và `frame_uid` coverage khớp;
- detector/recognizer output counts khớp;
- failure ledger không có unresolved failure;
- raw, normalized, bbox/crop và manifest đầy đủ;
- index truy ngược được về frame/line.

Quality gate cần qrels/benchmark được duyệt:

- detection/line recall;
- CER/WER và Vietnamese diacritic accuracy;
- exact accuracy cho số/mã/công thức;
- OCR-only Recall@K/MRR;
- ablation PP6, VietOCR, folded, n-gram, BGE và RRF.

Thiếu qrels thì status vẫn là `ENGINEERING_PROXY`, `quality_status=UNVALIDATED`.

## 10. Không làm

- Không gọi Kaggle `COMPLETE` là OCR quality COMPLETE.
- Không dùng `faiss_row`, parquet row hoặc Elasticsearch row làm identity.
- Không overwrite raw OCR bằng bản sửa/bỏ dấu.
- Không fuzzy mạnh trên số, mã hoặc công thức.
- Không gộp toàn bộ OCR của video thành một document duy nhất.
- Không xóa line khó; ghi `LOW_CONF/EMPTY` và giữ provenance.
- Không chạy PP5 full chỉ để làm fallback khi chưa có bằng chứng PP5 bổ sung recall.

## 11. Tài liệu tham khảo

- [PaddleOCR PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- [PaddleOCR PP-OCRv5 multilingual](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)
- [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md)
- [BGE-M3 repository](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/README.md) và [paper](https://arxiv.org/abs/2402.03216)
- [BGE reranker v2 m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Elasticsearch multi-fields](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/multi-fields), [multi-match](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-multi-match-query), [n-gram](https://www.elastic.co/docs/reference/elasticsearch/analysis-ngram-tokenizer.html), [RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [KPI: DeepSolo + PARSeq + text search](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_7)
- [FUSTAR: Vision + Elasticsearch](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_8)
- [AViSearch: DRRG/MMOCR + VietOCR + fuzzy search](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_18)
- [SnapSeek: PaddleOCR + Elasticsearch](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_23)
- [VISIONE OCR/ASR video retrieval](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321359/)
- [VinText Vietnamese scene text](https://minhhoai.net/papers/vintext_CVPR21.pdf)
- [PP-OCRv6 Vietnamese charset issue](https://github.com/PaddlePaddle/PaddleOCR/issues/18254)

