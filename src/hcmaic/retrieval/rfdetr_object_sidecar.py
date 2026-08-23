"""RF-DETR object sidecar adapter for engineering-proxy retrieval smoke tests.

The sidecar is read-only, immutable and fail-closed.  It is intentionally kept
separate from the validated-local object artifact loader so production paths do
not silently inherit remote-Kaggle provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract, build_channel_evidence
from hcmaic.retrieval.object_retrieval import ObjectArtifactError, ObjectUnavailableError

MANIFEST_NAME = "rfdetr_object_sidecar_manifest.json"
OBJECTS_NAME = "objects.jsonl"
FRAME_STATUS_NAME = "object_frame_status.jsonl"
FAILURE_LEDGER_NAME = "failure_ledger.json"
SCHEMA_VERSION = "hcmaic-rfdetr-object-sidecar-v1"
ARTIFACT_TYPE = "HCMAIC_RFDETR_XL_OBJECT_SIDECAR"
EXPECTED_IDENTITY = "frame_uid=video_id:source_frame_idx"
LOOKUP_INDEX_VERSION = "hcmaic-rfdetr-object-inverted-lookup-v1"
QUALITY_POSTPROCESS_VERSION = "hcmaic-rfdetr-object-quality-postprocess-v1"
OBJECT_ALIAS_MAP_VERSION = "hcmaic-object-alias-vi-coco-v3"

# Query-time aliases only.  They never rewrite the immutable sidecar labels
# or create label_canonical values.  Targets are filtered against the labels
# actually present in the loaded artifact before the catalog is exposed.
DEFAULT_OBJECT_ALIAS_MAP: dict[str, str] = {
    "man": "person",
    "men": "person",
    "people": "person",
    "persons": "person",
    "human": "person",
    "humans": "person",
    "adult": "person",
    "adults": "person",
    "individual": "person",
    "individuals": "person",
    "human being": "person",
    "human beings": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "woman": "person",
    "women": "person",
    "child": "person",
    "children": "person",
    "kid": "person",
    "kids": "person",
    "người": "person",
    "nguoi": "person",
    "con người": "person",
    "con nguoi": "person",
    "cars": "car",
    "automobile": "car",
    "automobiles": "car",
    "auto": "car",
    "autos": "car",
    "sedan": "car",
    "sedans": "car",
    "saloon": "car",
    "saloons": "car",
    "ô tô": "car",
    "oto": "car",
    "xe hơi": "car",
    "xe hoi": "car",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "motorcycles": "motorcycle",
    "xe gắn máy": "motorcycle",
    "xe gan may": "motorcycle",
    "xe máy": "motorcycle",
    "xe may": "motorcycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "cycle": "bicycle",
    "cycles": "bicycle",
    "pushbike": "bicycle",
    "pushbikes": "bicycle",
    "xe đạp": "bicycle",
    "xe dap": "bicycle",
    "buses": "bus",
    "xe buýt": "bus",
    "xe buyt": "bus",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
    "pickup truck": "truck",
    "pickup trucks": "truck",
    "xe tải": "truck",
    "xe tai": "truck",
    "planes": "airplane",
    "airplanes": "airplane",
    "plane": "airplane",
    "aeroplane": "airplane",
    "aeroplanes": "airplane",
    "aircraft": "airplane",
    "máy bay": "airplane",
    "may bay": "airplane",
    "apples": "apple",
    "táo": "apple",
    "tao": "apple",
    "bananas": "banana",
    "chuối": "banana",
    "chuoi": "banana",
    "baseball bats": "baseball bat",
    "gậy bóng chày": "baseball bat",
    "gay bong chay": "baseball bat",
    "baseball gloves": "baseball glove",
    "baseball mitt": "baseball glove",
    "baseball mitts": "baseball glove",
    "mitt": "baseball glove",
    "mitts": "baseball glove",
    "găng tay bóng chày": "baseball glove",
    "gang tay bong chay": "baseball glove",
    "bears": "bear",
    "gấu": "bear",
    "gau": "bear",
    "beds": "bed",
    "cot": "bed",
    "cots": "bed",
    "bunk bed": "bed",
    "bunk beds": "bed",
    "giường": "bed",
    "giuong": "bed",
    "benches": "bench",
    "ghế băng": "bench",
    "ghe bang": "bench",
    "boats": "boat",
    "rowboat": "boat",
    "rowboats": "boat",
    "sailboat": "boat",
    "sailboats": "boat",
    "speedboat": "boat",
    "speedboats": "boat",
    "thuyền": "boat",
    "thuyen": "boat",
    "books": "book",
    "sách": "book",
    "sach": "book",
    "bowls": "bowl",
    "bông cải xanh": "broccoli",
    "bong cai xanh": "broccoli",
    "carrots": "carrot",
    "cà rốt": "carrot",
    "ca rot": "carrot",
    "television": "tv",
    "televisions": "tv",
    "tivi": "tv",
    "ti vi": "tv",
    "ti-vi": "tv",
    "dogs": "dog",
    "puppy": "dog",
    "puppies": "dog",
    "chó": "dog",
    "cho": "dog",
    "cats": "cat",
    "kitty": "cat",
    "kitties": "cat",
    "feline": "cat",
    "mèo": "cat",
    "meo": "cat",
    "clocks": "clock",
    "đồng hồ": "clock",
    "dong ho": "clock",
    "couches": "couch",
    "sofa": "couch",
    "settee": "couch",
    "settees": "couch",
    "divan": "couch",
    "divans": "couch",
    "ghế sofa": "couch",
    "ghe sofa": "couch",
    "cows": "cow",
    "bò": "cow",
    "bo": "cow",
    "birds": "bird",
    "chim": "bird",
    "cups": "cup",
    "mug": "cup",
    "mugs": "cup",
    "tumbler": "cup",
    "tumblers": "cup",
    "cốc": "cup",
    "coc": "cup",
    "ly": "cup",
    "bottles": "bottle",
    "flask": "bottle",
    "flasks": "bottle",
    "chai": "bottle",
    "chairs": "chair",
    "armchair": "chair",
    "armchairs": "chair",
    "ghế": "chair",
    "ghe": "chair",
    "dining tables": "dining table",
    "table": "dining table",
    "tables": "dining table",
    "dinner table": "dining table",
    "dinner tables": "dining table",
    "bàn ăn": "dining table",
    "ban an": "dining table",
    "donuts": "donut",
    "doughnut": "donut",
    "doughnuts": "donut",
    "donut holes": "donut",
    "doughnut holes": "donut",
    "elephants": "elephant",
    "voi": "elephant",
    "fire hydrants": "fire hydrant",
    "vòi cứu hỏa": "fire hydrant",
    "voi cuu hoa": "fire hydrant",
    "forks": "fork",
    "eating fork": "fork",
    "eating forks": "fork",
    "dinner fork": "fork",
    "dinner forks": "fork",
    "nĩa": "fork",
    "nia": "fork",
    "frisbees": "frisbee",
    "flying disc": "frisbee",
    "flying discs": "frisbee",
    "flying disk": "frisbee",
    "flying disks": "frisbee",
    "đĩa bay": "frisbee",
    "dia bay": "frisbee",
    "giraffes": "giraffe",
    "hươu cao cổ": "giraffe",
    "huou cao co": "giraffe",
    "hair dryers": "hair drier",
    "hairdryer": "hair drier",
    "hairdryers": "hair drier",
    "máy sấy tóc": "hair drier",
    "may say toc": "hair drier",
    "handbags": "handbag",
    "purse": "handbag",
    "purses": "handbag",
    "clutch bag": "handbag",
    "clutch bags": "handbag",
    "túi xách": "handbag",
    "tui xach": "handbag",
    "horses": "horse",
    "pony": "horse",
    "ponies": "horse",
    "ngựa": "horse",
    "ngua": "horse",
    "hot dogs": "hot dog",
    "frankfurter": "hot dog",
    "frankfurters": "hot dog",
    "keyboards": "keyboard",
    "computer keyboard": "keyboard",
    "computer keyboards": "keyboard",
    "bàn phím": "keyboard",
    "ban phim": "keyboard",
    "kites": "kite",
    "diều": "kite",
    "dieu": "kite",
    "knives": "knife",
    "dao": "knife",
    "backpacks": "backpack",
    "rucksack": "backpack",
    "rucksacks": "backpack",
    "knapsack": "backpack",
    "knapsacks": "backpack",
    "schoolbag": "backpack",
    "schoolbags": "backpack",
    "ba lô": "backpack",
    "ba lo": "backpack",
    "microwaves": "microwave",
    "microwave oven": "microwave",
    "microwave ovens": "microwave",
    "lò vi sóng": "microwave",
    "lo vi song": "microwave",
    "mice": "mouse",
    "computer mouse": "mouse",
    "computer mice": "mouse",
    "chuột": "mouse",
    "chuot": "mouse",
    "oranges": "orange",
    "cam": "orange",
    "ovens": "oven",
    "cooker": "oven",
    "cookers": "oven",
    "lò nướng": "oven",
    "lo nuong": "oven",
    "parking meters": "parking meter",
    "pizza": "pizza",
    "pizzas": "pizza",
    "plant": "potted plant",
    "plants": "potted plant",
    "potted plants": "potted plant",
    "houseplant": "potted plant",
    "houseplants": "potted plant",
    "indoor plant": "potted plant",
    "indoor plants": "potted plant",
    "cây cảnh": "potted plant",
    "cay canh": "potted plant",
    "cây trong chậu": "potted plant",
    "cay trong chau": "potted plant",
    "fridge": "refrigerator",
    "refrigerators": "refrigerator",
    "tủ lạnh": "refrigerator",
    "tu lanh": "refrigerator",
    "remotes": "remote",
    "remote control": "remote",
    "remote controls": "remote",
    "điều khiển": "remote",
    "dieu khien": "remote",
    "sandwiches": "sandwich",
    "sub sandwich": "sandwich",
    "sub sandwiches": "sandwich",
    "hoagie": "sandwich",
    "hoagies": "sandwich",
    "panini": "sandwich",
    "paninis": "sandwich",
    "bánh sandwich": "sandwich",
    "banh sandwich": "sandwich",
    "scissors": "scissors",
    "shears": "scissors",
    "kéo": "scissors",
    "keo": "scissors",
    "sheep": "sheep",
    "cừu": "sheep",
    "cuu": "sheep",
    "sinks": "sink",
    "washbasin": "sink",
    "washbasins": "sink",
    "wash basin": "sink",
    "wash basins": "sink",
    "kitchen sink": "sink",
    "kitchen sinks": "sink",
    "bathroom sink": "sink",
    "bathroom sinks": "sink",
    "bồn rửa": "sink",
    "bon rua": "sink",
    "skateboards": "skateboard",
    "ván trượt": "skateboard",
    "van truot": "skateboard",
    "ski": "skis",
    "ván trượt tuyết": "skis",
    "van truot tuyet": "skis",
    "snowboards": "snowboard",
    "ván tuyết": "snowboard",
    "van tuyet": "snowboard",
    "spoons": "spoon",
    "teaspoon": "spoon",
    "teaspoons": "spoon",
    "tablespoon": "spoon",
    "tablespoons": "spoon",
    "thìa": "spoon",
    "thia": "spoon",
    "muỗng": "spoon",
    "muong": "spoon",
    "sports balls": "sports ball",
    "ball": "sports ball",
    "balls": "sports ball",
    "game ball": "sports ball",
    "game balls": "sports ball",
    "quả bóng": "sports ball",
    "qua bong": "sports ball",
    "stop signs": "stop sign",
    "biển báo dừng": "stop sign",
    "bien bao dung": "stop sign",
    "suitcases": "suitcase",
    "vali": "suitcase",
    "surfboards": "surfboard",
    "ván lướt sóng": "surfboard",
    "van luot song": "surfboard",
    "teddy bears": "teddy bear",
    "teddy": "teddy bear",
    "teddies": "teddy bear",
    "stuffed bear": "teddy bear",
    "stuffed bears": "teddy bear",
    "plush bear": "teddy bear",
    "plush bears": "teddy bear",
    "gấu bông": "teddy bear",
    "gau bong": "teddy bear",
    "tennis rackets": "tennis racket",
    "tennis racquet": "tennis racket",
    "tennis racquets": "tennis racket",
    "racquet": "tennis racket",
    "racquets": "tennis racket",
    "vợt tennis": "tennis racket",
    "vot tennis": "tennis racket",
    "ties": "tie",
    "necktie": "tie",
    "neckties": "tie",
    "neck tie": "tie",
    "neck ties": "tie",
    "cà vạt": "tie",
    "ca vat": "tie",
    "toasters": "toaster",
    "máy nướng bánh mì": "toaster",
    "may nuong banh mi": "toaster",
    "toilets": "toilet",
    "bồn cầu": "toilet",
    "bon cau": "toilet",
    "toothbrushes": "toothbrush",
    "bàn chải đánh răng": "toothbrush",
    "ban chai danh rang": "toothbrush",
    "traffic lights": "traffic light",
    "stoplight": "traffic light",
    "stoplights": "traffic light",
    "stop light": "traffic light",
    "stop lights": "traffic light",
    "signal light": "traffic light",
    "signal lights": "traffic light",
    "đèn giao thông": "traffic light",
    "den giao thong": "traffic light",
    "đèn tín hiệu": "traffic light",
    "den tin hieu": "traffic light",
    "trains": "train",
    "railway train": "train",
    "railway trains": "train",
    "subway train": "train",
    "subway trains": "train",
    "tàu hỏa": "train",
    "tau hoa": "train",
    "vases": "vase",
    "flower vase": "vase",
    "flower vases": "vase",
    "bình hoa": "vase",
    "binh hoa": "vase",
    "wine glasses": "wine glass",
    "goblet": "wine glass",
    "goblets": "wine glass",
    "ly rượu vang": "wine glass",
    "ly ruou vang": "wine glass",
    "zebras": "zebra",
    "ngựa vằn": "zebra",
    "ngua van": "zebra",
    "umbrellas": "umbrella",
    "parasol": "umbrella",
    "parasols": "umbrella",
    "ô": "umbrella",
    "dù": "umbrella",
    "du": "umbrella",
    "cellphones": "cell phone",
    "cell phone": "cell phone",
    "cell phones": "cell phone",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "mobile phones": "cell phone",
    "handset": "cell phone",
    "handsets": "cell phone",
    "smartphone": "cell phone",
    "smartphones": "cell phone",
    "điện thoại": "cell phone",
    "dien thoai": "cell phone",
    "điện thoại di động": "cell phone",
    "dien thoai di dong": "cell phone",
    "laptops": "laptop",
    "notebook computer": "laptop",
    "notebook computers": "laptop",
    "portable computer": "laptop",
    "portable computers": "laptop",
    "máy tính xách tay": "laptop",
    "may tinh xach tay": "laptop",
}

LOGGER = logging.getLogger(__name__)


class RfdetrObjectSidecarArtifactError(ObjectArtifactError):
    """Raised when the RF-DETR sidecar is malformed or hash-gated."""


class RfdetrObjectSidecarUnavailableError(ObjectUnavailableError):
    """Raised when the RF-DETR sidecar is not attached by policy."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RfdetrObjectSidecarArtifactError(f"cannot read JSON file {path}") from exc
    if not isinstance(payload, dict):
        raise RfdetrObjectSidecarArtifactError(f"JSON file must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RfdetrObjectSidecarArtifactError(
                    f"invalid JSONL row at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RfdetrObjectSidecarArtifactError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )
            rows.append(row)
    except OSError as exc:
        raise RfdetrObjectSidecarArtifactError(f"cannot read JSONL file {path}") from exc
    return rows


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RfdetrObjectSidecarArtifactError(message)


def _non_negative_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RfdetrObjectSidecarArtifactError(f"{context}: {field} must be a non-negative integer")
    return value


def _finite_float(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RfdetrObjectSidecarArtifactError(f"{context}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RfdetrObjectSidecarArtifactError(f"{context}: {field} must be finite")
    return result


def _required_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RfdetrObjectSidecarArtifactError(f"{context}: {field} must be a non-empty string")
    return value.strip()


def _normalized_raw_label(label: str) -> str:
    """Normalize only case and whitespace; do not translate or tokenize labels."""

    return " ".join(label.strip().casefold().split())


@dataclass(frozen=True)
class ParsedObjectQuery:
    """Parsed object lookup syntax without changing the sidecar label vocabulary."""

    raw_query: str
    label: str
    min_instances: int = 1


def parse_object_query(text: str) -> ParsedObjectQuery:
    """Parse an optional leading instance count from an object query.

    ``3 person`` means a frame must contain at least three detections of the
    raw label ``person``.  The parser deliberately does not translate labels
    or enable aliases; it only removes the numeric query prefix.
    """

    if not isinstance(text, str):
        raise TypeError("object query must be a string")
    raw_query = " ".join(text.strip().split())
    if not raw_query:
        return ParsedObjectQuery(raw_query="", label="", min_instances=1)

    match = re.fullmatch(r"(?P<count>\d+)\s+(?P<label>.+)", raw_query)
    if match:
        min_instances = int(match.group("count"))
        if min_instances < 1:
            raise ValueError("object query quantity must be >= 1")
        label = match.group("label")
    else:
        min_instances = 1
        label = raw_query
    normalized_label = _normalized_raw_label(label)
    if not normalized_label:
        raise ValueError("object query label must not be blank")
    return ParsedObjectQuery(
        raw_query=raw_query,
        label=normalized_label,
        min_instances=min_instances,
    )


def parse_object_query_list(text: str) -> tuple[ParsedObjectQuery, ...]:
    """Parse one or more quantity/object clauses joined with ``+``.

    ``2 people + 1 car`` is an AND query: a frame must satisfy both clauses.
    The parser keeps aliases unresolved so the adapter can attach the
    versioned alias-map provenance to each returned clause.
    """

    if not isinstance(text, str):
        raise TypeError("object query must be a string")
    raw_query = " ".join(text.strip().split())
    if not raw_query:
        return ()
    parts = [part.strip() for part in re.split(r"\s*(?:\+|;)\s*", raw_query)]
    if any(not part for part in parts):
        raise ValueError("object query contains an empty clause")
    return tuple(parse_object_query(part) for part in parts)


def _bbox(value: Any, field: str, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise RfdetrObjectSidecarArtifactError(f"{context}: {field} must be [x1,y1,x2,y2]")
    values = tuple(_finite_float(item, field, context) for item in value)
    if any(item < 0.0 for item in values):
        raise RfdetrObjectSidecarArtifactError(
            f"{context}: {field} coordinates must be non-negative"
        )
    if values[2] < values[0] or values[3] < values[1]:
        raise RfdetrObjectSidecarArtifactError(f"{context}: {field} coordinates are inverted")
    return values


@dataclass(frozen=True)
class ObjectQualityPostprocessConfig:
    """Deterministic quality filters applied to raw same-label boxes.

    NMS follows the standard class-wise detector postprocess convention.  The
    containment threshold is an additional conservative rule for duplicate
    boxes where a small lower-confidence box is almost entirely inside a
    higher-confidence box but its IoU is too small to trigger NMS.
    """

    nms_iou_threshold: float = 0.5
    containment_threshold: float = 0.8
    version: str = QUALITY_POSTPROCESS_VERSION

    def __post_init__(self) -> None:
        for field_name in ("nms_iou_threshold", "containment_threshold"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nms_iou_threshold": float(self.nms_iou_threshold),
            "containment_threshold": float(self.containment_threshold),
        }


DEFAULT_OBJECT_QUALITY_POSTPROCESS_CONFIG = ObjectQualityPostprocessConfig()


@dataclass(frozen=True)
class ObjectQualityPostprocessResult:
    instances: tuple[dict[str, Any], ...]
    suppressed_nms_count: int
    suppressed_containment_count: int

    @property
    def suppressed_count(self) -> int:
        return self.suppressed_nms_count + self.suppressed_containment_count


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = _box_intersection_area(left, right)
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def _candidate_containment(
    candidate: tuple[float, float, float, float],
    kept: tuple[float, float, float, float],
) -> float:
    candidate_area = _box_area(candidate)
    return _box_intersection_area(candidate, kept) / candidate_area if candidate_area > 0.0 else 0.0


def quality_postprocess_instances(
    instances: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    config: ObjectQualityPostprocessConfig = DEFAULT_OBJECT_QUALITY_POSTPROCESS_CONFIG,
) -> ObjectQualityPostprocessResult:
    """Apply deterministic same-label NMS plus duplicate containment filtering.

    The function returns copied selected instances and never mutates the raw
    sidecar payload.  Inputs are expected to have already passed the strict
    sidecar schema validator; the small amount of geometry validation here
    keeps the helper safe for unit-level use as well.
    """

    if not isinstance(config, ObjectQualityPostprocessConfig):
        raise TypeError("config must be ObjectQualityPostprocessConfig")

    ordered: list[tuple[dict[str, Any], tuple[float, float, float, float]]] = []
    for instance in instances:
        if not isinstance(instance, Mapping):
            raise TypeError("object instance must be a mapping")
        payload = dict(instance)
        try:
            bbox_value = payload["bbox"]
            confidence_value = payload["confidence"]
        except KeyError as exc:
            raise ValueError(f"object instance missing {exc.args[0]}") from exc
        if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
            raise ValueError("object instance bbox must contain four coordinates")
        try:
            bbox = tuple(float(value) for value in bbox_value)
            confidence = float(confidence_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("object instance bbox/confidence must be numeric") from exc
        if any(not math.isfinite(value) for value in (*bbox, confidence)):
            raise ValueError("object instance bbox/confidence must be finite")
        if any(value < 0.0 for value in bbox) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("object instance bbox must be non-negative and non-inverted")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("object instance confidence must be in [0, 1]")
        ordered.append((payload, bbox))

    if len(ordered) <= 1:
        return ObjectQualityPostprocessResult(
            instances=tuple(dict(payload) for payload, _ in ordered),
            suppressed_nms_count=0,
            suppressed_containment_count=0,
        )

    ordered.sort(
        key=lambda item: (
            -float(item[0]["confidence"]),
            int(item[0].get("box_index", 0)),
            int(item[0].get("record_index", 0)),
        )
    )

    kept: list[tuple[dict[str, Any], tuple[float, float, float, float]]] = []
    suppressed_nms_count = 0
    suppressed_containment_count = 0
    for candidate, candidate_box in ordered:
        suppressed = False
        for _, kept_box in kept:
            if _box_iou(candidate_box, kept_box) > float(config.nms_iou_threshold):
                suppressed_nms_count += 1
                suppressed = True
                break
            if _candidate_containment(candidate_box, kept_box) >= float(
                config.containment_threshold
            ):
                suppressed_containment_count += 1
                suppressed = True
                break
        if not suppressed:
            kept.append((candidate, candidate_box))

    return ObjectQualityPostprocessResult(
        instances=tuple(dict(candidate) for candidate, _ in kept),
        suppressed_nms_count=suppressed_nms_count,
        suppressed_containment_count=suppressed_containment_count,
    )


def _identity_hash(frame_uids: set[str]) -> str:
    payload = "".join(f"{frame_uid}\n" for frame_uid in sorted(frame_uids)).encode("utf-8")
    return _sha256_bytes(payload)


def _first_nonblank(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _instance_payload(
    instance: Mapping[str, Any], *, record_index: int, frame_uid: str
) -> dict[str, Any]:
    payload = dict(instance)
    payload["record_index"] = record_index
    payload["frame_uid"] = frame_uid
    return payload


@dataclass(frozen=True)
class _GroupedHit:
    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    normalized_label: str
    raw_labels: tuple[str, ...]
    confidence: float
    raw_confidence: float
    bbox: tuple[float, float, float, float] | None
    raw_bbox: tuple[float, float, float, float] | None
    records: tuple[dict[str, Any], ...]
    instances: tuple[dict[str, Any], ...]
    quality_instances: tuple[dict[str, Any], ...]
    quality_suppressed_nms_count: int
    quality_suppressed_containment_count: int
    source_shard_ids: tuple[str, ...]
    frame_status: dict[str, Any] | None
    provider: str
    revision: str
    provider_execution: str
    label_source: str
    model_id: str
    model_weights_sha256: str
    keyframe_paths: tuple[str, ...]


@dataclass(frozen=True)
class RfdetrObjectSidecarArtifact:
    artifact_dir: Path
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    frame_status: dict[str, dict[str, Any]]
    failure_ledger: dict[str, Any]
    groups: tuple[_GroupedHit, ...]
    quality_config: ObjectQualityPostprocessConfig
    manifest_sha256: str
    objects_sha256: str
    frame_status_sha256: str
    failure_ledger_sha256: str


def _validate_frame_status_row(row: Mapping[str, Any], context: str) -> tuple[str, str, int]:
    required = (
        "detection_count",
        "error",
        "frame_uid",
        "height",
        "inference_ms",
        "review_image",
        "source_frame_idx",
        "source_shard_id",
        "status",
        "video_id",
        "width",
    )
    missing = [field for field in required if field not in row]
    _require(not missing, f"{context}: frame status missing fields: {missing}")
    frame_uid = _required_string(row["frame_uid"], "frame_uid", context)
    video_id = _required_string(row["video_id"], "video_id", context)
    source_frame_idx = _non_negative_int(row["source_frame_idx"], "source_frame_idx", context)
    _require(
        frame_uid == f"{video_id}:{source_frame_idx}",
        f"{context}: frame_uid must match video_id:source_frame_idx",
    )
    _require(row["status"] == "OK", f"{context}: status must be OK")
    _non_negative_int(row["detection_count"], "detection_count", context)
    _non_negative_int(row["height"], "height", context)
    _non_negative_int(row["width"], "width", context)
    _require(
        _finite_float(row["inference_ms"], "inference_ms", context) >= 0.0,
        f"{context}: inference_ms must be non-negative",
    )
    _required_string(row["source_shard_id"], "source_shard_id", context)
    for field in ("error", "review_image"):
        _require(
            row[field] is None or isinstance(row[field], str),
            f"{context}: {field} must be a string or null",
        )
    return frame_uid, video_id, source_frame_idx


def _validate_object_record(
    record: Mapping[str, Any], *, record_index: int, manifest: Mapping[str, Any]
) -> tuple[str, str, int, int, str, float, int]:
    context = f"object record {record_index + 1}"
    required = (
        "bbox",
        "confidence",
        "frame_uid",
        "instance_count",
        "instances",
        "keyframe_path",
        "label",
        "label_canonical",
        "label_raw",
        "label_raw_variants",
        "label_source",
        "model_id",
        "model_weights_sha256",
        "provider",
        "provider_execution",
        "revision",
        "source_frame_idx",
        "source_shard_ids",
        "timestamp_ms",
        "video_id",
    )
    missing = [field for field in required if field not in record]
    _require(not missing, f"{context}: missing fields: {missing}")

    frame_uid = _required_string(record["frame_uid"], "frame_uid", context)
    video_id = _required_string(record["video_id"], "video_id", context)
    source_frame_idx = _non_negative_int(record["source_frame_idx"], "source_frame_idx", context)
    timestamp_ms = _non_negative_int(record["timestamp_ms"], "timestamp_ms", context)
    _require(
        frame_uid == f"{video_id}:{source_frame_idx}",
        f"{context}: frame_uid must match video_id:source_frame_idx",
    )

    label_raw = _required_string(record["label_raw"], "label_raw", context)
    _require(record["label"] == label_raw, f"{context}: label must equal label_raw")
    _require(record["label_canonical"] is None, f"{context}: label_canonical must remain null")
    variants = record["label_raw_variants"]
    _require(
        isinstance(variants, list)
        and variants
        and all(isinstance(item, str) and item.strip() for item in variants),
        f"{context}: label_raw_variants must be a non-empty string array",
    )
    _require(label_raw in variants, f"{context}: label_raw must be retained in label_raw_variants")
    _require(
        record["label_source"] == "rfdetr_coco",
        f"{context}: label_source must remain rfdetr_coco",
    )
    _require(record["provider"] == "rfdetr_coco", f"{context}: provider must remain rfdetr_coco")
    _require(
        record["provider_execution"] == "remote-kaggle",
        f"{context}: provider_execution must remain remote-kaggle",
    )
    revision = _required_string(record["revision"], "revision", context)
    model_id = _required_string(record["model_id"], "model_id", context)
    weights_hash = record["model_weights_sha256"]
    _require(_valid_sha256(weights_hash), f"{context}: model_weights_sha256 must be sha256")
    model_contract = manifest.get("model_contract")
    _require(isinstance(model_contract, Mapping), "model_contract must be present")
    _require(model_id == model_contract.get("model_id"), f"{context}: model_id mismatch")
    _require(
        weights_hash == model_contract.get("model_weights_sha256"),
        f"{context}: model_weights_sha256 mismatch",
    )
    _require(
        revision == f"{model_id}@{weights_hash}",
        f"{context}: revision/model provenance mismatch",
    )

    confidence = _finite_float(record["confidence"], "confidence", context)
    _require(0.0 <= confidence <= 1.0, f"{context}: confidence must be in [0, 1]")
    _bbox(record["bbox"], "bbox", context)
    keyframe_path = _required_string(record["keyframe_path"], "keyframe_path", context)
    relative_path = Path(keyframe_path)
    _require(
        not relative_path.is_absolute() and ".." not in relative_path.parts,
        f"{context}: keyframe_path must remain relative and traversal-free",
    )
    source_shard_ids = record["source_shard_ids"]
    _require(
        isinstance(source_shard_ids, list)
        and source_shard_ids
        and all(isinstance(item, str) and item.strip() for item in source_shard_ids),
        f"{context}: source_shard_ids must be a non-empty string array",
    )
    instances = record["instances"]
    instance_count = _non_negative_int(record["instance_count"], "instance_count", context)
    _require(instance_count > 0, f"{context}: instance_count must be positive")
    _require(
        isinstance(instances, list) and len(instances) == instance_count,
        f"{context}: instance_count must equal len(instances)",
    )
    for instance_index, instance in enumerate(instances, start=1):
        instance_context = f"{context} instance {instance_index}"
        _require(isinstance(instance, Mapping), f"{instance_context}: instance must be an object")
        for field in ("box_index", "class_id", "confidence", "bbox"):
            _require(field in instance, f"{instance_context}: missing {field}")
        _non_negative_int(instance["box_index"], "box_index", instance_context)
        _non_negative_int(instance["class_id"], "class_id", instance_context)
        instance_confidence = _finite_float(instance["confidence"], "confidence", instance_context)
        _require(
            0.0 <= instance_confidence <= 1.0,
            f"{instance_context}: confidence must be in [0, 1]",
        )
        _bbox(instance["bbox"], "bbox", instance_context)
    return (
        frame_uid,
        video_id,
        source_frame_idx,
        timestamp_ms,
        label_raw,
        confidence,
        instance_count,
    )


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    objects_path: Path,
    frame_status_path: Path,
    failure_ledger_path: Path,
    objects_sha256: str,
    frame_status_sha256: str,
    failure_ledger_sha256: str,
    n_records: int,
    n_unique_frames: int,
    n_detection_instances: int,
    n_frames_with_detections: int,
    n_zero_detection_frames: int,
    duplicate_group_count: int,
    duplicate_extra_instance_count: int,
    frames_with_multiple_labels: int,
    recomputed_identity_hash: str,
) -> None:
    _require(manifest.get("artifact_type") == ARTIFACT_TYPE, "unsupported RF-DETR artifact type")
    _require(
        manifest.get("manifest_schema_version") == SCHEMA_VERSION,
        "unsupported RF-DETR manifest schema version",
    )
    _require(manifest.get("identity") == EXPECTED_IDENTITY, "RF-DETR identity contract mismatch")
    _require(_valid_sha256(manifest.get("identity_hash")), "identity_hash must be sha256")
    _require(manifest.get("records") == OBJECTS_NAME, "records path must be objects.jsonl")
    _require(manifest.get("frame_status") == FRAME_STATUS_NAME, "frame_status path mismatch")
    _require(manifest.get("failure_ledger") == FAILURE_LEDGER_NAME, "failure_ledger path mismatch")
    _require(manifest.get("status") == "ENGINEERING_SIDECAR_COMPLETE", "sidecar status mismatch")
    _require(
        manifest.get("quality_status") == "UNVALIDATED", "quality_status must remain UNVALIDATED"
    )
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, Mapping), "provenance must be present")
    _require(
        provenance.get("execution_status") == "ENGINEERING_PROXY",
        "provenance execution_status must remain ENGINEERING_PROXY",
    )
    _require(
        provenance.get("source_execution") == "remote-kaggle",
        "provenance source_execution must be remote-kaggle",
    )
    _require(
        provenance.get("raw_labels_preserved") is True,
        "raw COCO labels must be preserved",
    )
    gate = manifest.get("promotion_gate")
    _require(isinstance(gate, dict), "promotion_gate must be present")
    _require(gate.get("zero_unresolved_failures") is True, "zero unresolved failures required")
    _require(gate.get("all_source_shards_complete") is True, "source shards must be complete")
    _require(
        gate.get("all_source_frame_status_reconciled") is True,
        "frame status reconciliation required",
    )
    _require(gate.get("global_frame_uid_unique") is True, "global frame_uid uniqueness required")
    _require(manifest.get("n_records") == n_records, "n_records mismatch")
    _require(manifest.get("n_unique_frames") == n_unique_frames, "n_unique_frames mismatch")
    _require(
        manifest.get("n_detection_instances") == n_detection_instances,
        "n_detection_instances mismatch",
    )
    _require(
        manifest.get("n_frames_with_detections") == n_frames_with_detections,
        "n_frames_with_detections mismatch",
    )
    _require(
        manifest.get("n_zero_detection_frames") == n_zero_detection_frames,
        "n_zero_detection_frames mismatch",
    )
    _require(
        manifest.get("duplicate_group_count") == duplicate_group_count,
        "duplicate_group_count mismatch",
    )
    _require(
        manifest.get("duplicate_extra_instance_count") == duplicate_extra_instance_count,
        "duplicate_extra_instance_count mismatch",
    )
    _require(
        manifest.get("frames_with_multiple_labels") == frames_with_multiple_labels,
        "frames_with_multiple_labels mismatch",
    )
    _require(
        manifest.get("identity_hash") == recomputed_identity_hash,
        "identity_hash mismatch",
    )
    _require(
        manifest.get("source_shard_count") == len(manifest.get("source_shards", [])),
        "source_shard_count mismatch",
    )
    _require(manifest.get("source_shard_count", 0) >= 1, "source_shard_count must be positive")
    output_hashes = manifest.get("output_hashes")
    _require(isinstance(output_hashes, dict), "output_hashes must be present")
    for name in (OBJECTS_NAME, FRAME_STATUS_NAME, FAILURE_LEDGER_NAME):
        _require(_valid_sha256(output_hashes.get(name)), f"{name} declared hash must be sha256")
    _require(output_hashes.get("objects.jsonl") == objects_sha256, "objects.jsonl hash mismatch")
    _require(
        output_hashes.get("object_frame_status.jsonl") == frame_status_sha256,
        "object_frame_status.jsonl hash mismatch",
    )
    _require(
        output_hashes.get("failure_ledger.json") == failure_ledger_sha256,
        "failure_ledger.json hash mismatch",
    )

    failure_ledger = _read_json(failure_ledger_path)
    _require(failure_ledger.get("status") == "GREEN", "failure ledger must remain GREEN")
    _require(
        failure_ledger.get("schema_version") == SCHEMA_VERSION,
        "failure ledger schema version mismatch",
    )
    source_unresolved_count = failure_ledger.get("source_unresolved_count")
    _require(
        _non_negative_int(source_unresolved_count, "source_unresolved_count", "failure ledger")
        == 0,
        "source unresolved count must be zero",
    )
    unresolved = failure_ledger.get("unresolved")
    _require(isinstance(unresolved, list), "failure ledger unresolved must be a list")
    _require(not unresolved, "failure ledger unresolved must be empty")
    if "unresolved_count" not in failure_ledger:
        LOGGER.warning(
            "RF-DETR sidecar compatibility: aggregate failure_ledger.json omits "
            "unresolved_count; unresolved=[] and source_unresolved_count=0, so the "
            "loader treats the missing field as compatible zero."
        )
    else:
        unresolved_count = _non_negative_int(
            failure_ledger.get("unresolved_count"), "unresolved_count", "failure ledger"
        )
        _require(unresolved_count == len(unresolved), "unresolved count does not match entries")

    ledger_entries = failure_ledger.get("source_ledgers")
    _require(isinstance(ledger_entries, list) and ledger_entries, "source_ledgers must be present")
    ledger_by_shard: dict[str, Mapping[str, Any]] = {}
    for entry in ledger_entries:
        _require(isinstance(entry, Mapping), "source ledger entries must be objects")
        shard_id = _required_string(entry.get("shard_id"), "shard_id", "source ledger")
        _require(shard_id not in ledger_by_shard, "source ledger shard_id must be unique")
        _require(_valid_sha256(entry.get("sha256")), "source ledger sha256 must be sha256")
        _require(
            _non_negative_int(entry.get("unresolved_count"), "unresolved_count", "source ledger")
            == 0,
            "source ledger unresolved_count must be zero",
        )
        ledger_by_shard[shard_id] = entry

    source_shards = manifest.get("source_shards")
    _require(isinstance(source_shards, list) and source_shards, "source_shards must be present")
    source_shard_ids: set[str] = set()
    for shard in source_shards:
        _require(isinstance(shard, Mapping), "source_shards entries must be objects")
        shard_id = _required_string(shard.get("shard_id"), "shard_id", "source shard")
        _require(shard_id not in source_shard_ids, "source shard id must be unique")
        source_shard_ids.add(shard_id)
        _require(
            _non_negative_int(
                shard.get("detection_row_count"), "detection_row_count", "source shard"
            )
            >= 0,
            "source shard detection row count required",
        )
        _require(
            _non_negative_int(
                shard.get("selected_frame_count"), "selected_frame_count", "source shard"
            )
            >= 0,
            "source shard selected frame count required",
        )
        _require(
            _non_negative_int(shard.get("review_image_count"), "review_image_count", "source shard")
            >= 0,
            "source shard review image count required",
        )
        _require(
            _valid_sha256(shard.get("source_detection_sha256")),
            "source_detection_sha256 must be sha256",
        )
        _require(
            _valid_sha256(shard.get("source_manifest_sha256")),
            "source_manifest_sha256 must be sha256",
        )
        _require(
            _valid_sha256(shard.get("source_status_sha256")), "source_status_sha256 must be sha256"
        )
        source_ledger = ledger_by_shard.get(shard_id)
        _require(source_ledger is not None, f"source failure ledger missing shard {shard_id}")
        _require(
            _valid_sha256(shard.get("source_failure_ledger_sha256")),
            "source_failure_ledger_sha256 must be sha256",
        )
        _require(
            shard.get("source_failure_ledger_sha256") == source_ledger.get("sha256"),
            "source failure ledger hash mismatch",
        )

    source_selected_frames = sum(
        int(shard.get("selected_frame_count", 0)) for shard in source_shards
    )
    source_detection_rows = sum(int(shard.get("detection_row_count", 0)) for shard in source_shards)
    _require(source_selected_frames == n_unique_frames, "selected frame count mismatch")
    _require(source_detection_rows == n_detection_instances, "source detection row count mismatch")
    _require(source_shard_ids == set(ledger_by_shard), "source shard/ledger set mismatch")

    _ = manifest_path, objects_path  # keep explicit read-only contract in scope


def _group_records(
    records: list[dict[str, Any]],
    frame_status: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    *,
    quality_config: ObjectQualityPostprocessConfig,
) -> tuple[tuple[_GroupedHit, ...], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    frames_with_labels: dict[str, set[str]] = defaultdict(set)
    detection_instances_by_frame: Counter[str] = Counter()
    for record_index, record in enumerate(records):
        frame_uid, video_id, source_frame_idx, timestamp_ms, label, confidence, instance_count = (
            _validate_object_record(record, record_index=record_index, manifest=manifest)
        )
        normalized_label = _normalized_raw_label(label)
        _require(
            normalized_label,
            f"object record {record_index + 1}: normalized label must not be blank",
        )
        _require(
            frame_uid in frame_status,
            f"object record {record_index + 1}: frame status is missing",
        )
        detection_instances_by_frame[frame_uid] += instance_count
        frames_with_labels[frame_uid].add(normalized_label)
        group = groups.setdefault(
            (frame_uid, normalized_label),
            {
                "frame_uid": frame_uid,
                "video_id": video_id,
                "video_filename": str(record.get("video_filename") or f"{video_id}.mp4"),
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": timestamp_ms,
                "normalized_label": normalized_label,
                "raw_labels": [],
                "raw_confidence": 0.0,
                "raw_bbox": None,
                "records": [],
                "instances": [],
                "source_shard_ids": set(),
                "provider": str(record.get("provider")),
                "revision": str(record.get("revision")),
                "provider_execution": str(record.get("provider_execution")),
                "label_source": str(record.get("label_source")),
                "model_id": str(record.get("model_id")),
                "model_weights_sha256": str(record.get("model_weights_sha256")),
                "keyframe_paths": set(),
            },
        )
        group["raw_labels"].append(label)
        group["records"].append(record)
        group["source_shard_ids"].update(
            str(item) for item in (record.get("source_shard_ids") or [])
        )
        group["keyframe_paths"].add(str(record.get("keyframe_path") or ""))
        if confidence > float(group["raw_confidence"]):
            group["raw_confidence"] = confidence
            group["raw_bbox"] = tuple(record["bbox"])
        for instance in record["instances"]:
            group["instances"].append(
                _instance_payload(instance, record_index=record_index, frame_uid=frame_uid)
            )

    duplicate_group_count = 0
    grouped_hits: list[_GroupedHit] = []
    for payload in groups.values():
        if len(payload["instances"]) > 1:
            duplicate_group_count += 1
        ordered_records = tuple(
            payload["records"]
            if len(payload["records"]) <= 1
            else sorted(
                payload["records"],
                key=lambda item: (
                    tuple(str(value) for value in item.get("source_shard_ids", [])),
                    -float(item["confidence"]),
                    str(item.get("keyframe_path", "")),
                ),
            )
        )
        ordered_instances = tuple(
            payload["instances"]
            if len(payload["instances"]) <= 1
            else sorted(
                payload["instances"],
                key=lambda item: (
                    -float(item["confidence"]),
                    int(item["box_index"]),
                    int(item.get("record_index", 0)),
                ),
            )
        )
        quality_result = quality_postprocess_instances(
            ordered_instances,
            config=quality_config,
        )
        quality_instances = quality_result.instances
        quality_bbox = (
            tuple(float(value) for value in quality_instances[0]["bbox"])
            if quality_instances
            else None
        )
        quality_confidence = float(quality_instances[0]["confidence"]) if quality_instances else 0.0
        grouped_hits.append(
            _GroupedHit(
                frame_uid=payload["frame_uid"],
                video_id=payload["video_id"],
                video_filename=payload["video_filename"],
                source_frame_idx=payload["source_frame_idx"],
                timestamp_ms=payload["timestamp_ms"],
                normalized_label=payload["normalized_label"],
                raw_labels=tuple(sorted(set(payload["raw_labels"]))),
                confidence=quality_confidence,
                raw_confidence=float(payload["raw_confidence"]),
                bbox=quality_bbox,
                raw_bbox=payload["raw_bbox"],
                records=ordered_records,
                instances=ordered_instances,
                quality_instances=quality_instances,
                quality_suppressed_nms_count=quality_result.suppressed_nms_count,
                quality_suppressed_containment_count=quality_result.suppressed_containment_count,
                source_shard_ids=tuple(
                    sorted(item for item in payload["source_shard_ids"] if item)
                ),
                frame_status=dict(frame_status.get(payload["frame_uid"], {})) or None,
                provider=payload["provider"],
                revision=payload["revision"],
                provider_execution=payload["provider_execution"],
                label_source=payload["label_source"],
                model_id=payload["model_id"],
                model_weights_sha256=payload["model_weights_sha256"],
                keyframe_paths=tuple(sorted(item for item in payload["keyframe_paths"] if item)),
            )
        )
    grouped_hits.sort(
        key=lambda item: (
            item.frame_uid,
            item.normalized_label,
            item.source_frame_idx,
        )
    )
    status_uids = set(frame_status)
    object_uids = set(detection_instances_by_frame)
    positive_status_uids = {
        frame_uid for frame_uid, row in frame_status.items() if int(row["detection_count"]) > 0
    }
    _require(object_uids == positive_status_uids, "object/status identity coverage mismatch")
    for frame_uid, row in frame_status.items():
        _require(
            detection_instances_by_frame[frame_uid] == int(row["detection_count"]),
            f"{frame_uid}: status detection_count mismatch",
        )
    n_records = len(records)
    n_unique_frames = len(status_uids)
    n_detection_instances = sum(detection_instances_by_frame.values())
    n_frames_with_detections = sum(
        1 for payload in frame_status.values() if int(payload.get("detection_count", 0)) > 0
    )
    n_zero_detection_frames = sum(
        1 for payload in frame_status.values() if int(payload.get("detection_count", 0)) == 0
    )
    frames_with_multiple_labels = len(
        {frame_uid for frame_uid, labels in frames_with_labels.items() if len(labels) > 1}
    )
    counts = {
        "n_records": n_records,
        "n_unique_frames": n_unique_frames,
        "n_detection_instances": n_detection_instances,
        "n_frames_with_detections": n_frames_with_detections,
        "n_zero_detection_frames": n_zero_detection_frames,
        "duplicate_group_count": duplicate_group_count,
        "duplicate_extra_instance_count": n_detection_instances - n_records,
        "frames_with_multiple_labels": frames_with_multiple_labels,
    }
    return tuple(grouped_hits), counts


def load_rfdetr_object_sidecar(
    artifact_dir: Path,
    *,
    allow_engineering_proxy: bool = False,
    expected_frame_uids: set[str] | None = None,
    quality_config: ObjectQualityPostprocessConfig | None = None,
) -> RfdetrObjectSidecarArtifact:
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    if not allow_engineering_proxy:
        raise RfdetrObjectSidecarUnavailableError("engineering_proxy_disabled_by_policy")
    if quality_config is None:
        quality_config = DEFAULT_OBJECT_QUALITY_POSTPROCESS_CONFIG
    if not isinstance(quality_config, ObjectQualityPostprocessConfig):
        raise TypeError("quality_config must be ObjectQualityPostprocessConfig")

    manifest_path = artifact_dir / MANIFEST_NAME
    objects_path = artifact_dir / OBJECTS_NAME
    frame_status_path = artifact_dir / FRAME_STATUS_NAME
    failure_ledger_path = artifact_dir / FAILURE_LEDGER_NAME
    if not manifest_path.is_file():
        raise RfdetrObjectSidecarUnavailableError(f"missing RF-DETR manifest: {manifest_path}")
    if not objects_path.is_file():
        raise RfdetrObjectSidecarUnavailableError(f"missing RF-DETR objects: {objects_path}")
    if not frame_status_path.is_file():
        raise RfdetrObjectSidecarUnavailableError(
            f"missing RF-DETR frame status: {frame_status_path}"
        )
    if not failure_ledger_path.is_file():
        raise RfdetrObjectSidecarUnavailableError(
            f"missing RF-DETR failure ledger: {failure_ledger_path}"
        )

    manifest = _read_json(manifest_path)
    objects = _read_jsonl(objects_path)
    frame_rows = _read_jsonl(frame_status_path)
    failure_ledger = _read_json(failure_ledger_path)
    manifest_sha256 = _sha256_path(manifest_path)
    objects_sha256 = _sha256_path(objects_path)
    frame_status_sha256 = _sha256_path(frame_status_path)
    failure_ledger_sha256 = _sha256_path(failure_ledger_path)

    frame_status: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(frame_rows, start=1):
        frame_uid, _, _ = _validate_frame_status_row(row, f"frame status row {row_number}")
        _require(frame_uid not in frame_status, "frame status must have unique frame_uid values")
        frame_status[frame_uid] = row
    if expected_frame_uids is not None:
        expected = {str(frame_uid) for frame_uid in expected_frame_uids}
        _require(
            set(frame_status) == expected,
            "canonical frame_uid coverage mismatch",
        )

    grouped_hits, counts = _group_records(
        objects,
        frame_status,
        manifest,
        quality_config=quality_config,
    )
    _require(
        counts["n_frames_with_detections"] + counts["n_zero_detection_frames"]
        == counts["n_unique_frames"],
        "frame status coverage mismatch",
    )
    _validate_manifest(
        manifest,
        manifest_path=manifest_path,
        objects_path=objects_path,
        frame_status_path=frame_status_path,
        failure_ledger_path=failure_ledger_path,
        objects_sha256=objects_sha256,
        frame_status_sha256=frame_status_sha256,
        failure_ledger_sha256=failure_ledger_sha256,
        recomputed_identity_hash=_identity_hash(set(frame_status)),
        **counts,
    )
    return RfdetrObjectSidecarArtifact(
        artifact_dir=artifact_dir,
        manifest=manifest,
        records=tuple(objects),
        frame_status=frame_status,
        failure_ledger=failure_ledger,
        groups=grouped_hits,
        quality_config=quality_config,
        manifest_sha256=manifest_sha256,
        objects_sha256=objects_sha256,
        frame_status_sha256=frame_status_sha256,
        failure_ledger_sha256=failure_ledger_sha256,
    )


class RfdetrObjectSidecarAdapter:
    """Deterministic raw-label retrieval over a validated sidecar.

    The adapter owns only an in-memory inverted lookup.  It never changes the
    sidecar rows, canonical catalog, or visual indexes.  Matching is exact
    after case/whitespace normalization.  Aliases are never enabled
    implicitly; callers must provide an explicitly versioned alias map when a
    deployment has an approved vocabulary mapping.
    """

    def __init__(
        self,
        artifact: RfdetrObjectSidecarArtifact,
        *,
        alias_map: Mapping[str, str] | None = None,
        alias_map_version: str | None = None,
    ) -> None:
        self.artifact = artifact
        self.quality_config = artifact.quality_config
        self._groups = artifact.groups
        self._postings: dict[str, list[int]] = defaultdict(list)
        for group_index, group in enumerate(self._groups):
            self._postings[group.normalized_label].append(group_index)
        for postings in self._postings.values():
            postings.sort(
                key=lambda group_index: (
                    self._groups[group_index].video_id,
                    self._groups[group_index].source_frame_idx,
                    self._groups[group_index].frame_uid,
                )
            )
        self._ranked_postings: dict[str, tuple[int, ...]] = {
            label: tuple(
                sorted(
                    indexes,
                    key=lambda group_index: (
                        -float(self._groups[group_index].confidence),
                        self._groups[group_index].video_id,
                        self._groups[group_index].source_frame_idx,
                        self._groups[group_index].frame_uid,
                        self._groups[group_index].normalized_label,
                    ),
                )
            )
            for label, indexes in self._postings.items()
        }

        if alias_map is None:
            if alias_map_version is not None:
                raise ValueError("alias_map_version requires an explicit alias_map")
            self._alias_map = {}
            self._alias_map_version = None
        else:
            if not isinstance(alias_map_version, str) or not alias_map_version.strip():
                raise ValueError("an explicit alias_map requires alias_map_version")
            normalized_aliases: dict[str, str] = {}
            for alias, target in alias_map.items():
                if not isinstance(alias, str) or not isinstance(target, str):
                    raise ValueError("alias_map keys and values must be strings")
                normalized_alias = _normalized_raw_label(alias)
                normalized_target = _normalized_raw_label(target)
                if not normalized_alias or not normalized_target:
                    raise ValueError("alias_map keys and values must not be blank")
                if normalized_target not in self._postings:
                    raise ValueError(
                        f"alias_map target is absent from raw labels: {normalized_target!r}"
                    )
                normalized_aliases[normalized_alias] = normalized_target
            self._alias_map = dict(sorted(normalized_aliases.items()))
            self._alias_map_version = alias_map_version.strip()

        postings_payload = {
            label: [self._groups[index].frame_uid for index in indexes]
            for label, indexes in sorted(self._postings.items())
        }
        postings_bytes = json.dumps(
            postings_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        match_mode = (
            "exact_normalized_label_raw_with_aliases"
            if self._alias_map
            else "exact_normalized_label_raw"
        )
        index_payload = {
            "index_version": LOOKUP_INDEX_VERSION,
            "match": match_mode,
            "alias_map": self._alias_map,
            "alias_map_version": self._alias_map_version,
            "quality_postprocess": self.quality_config.as_dict(),
            "postings": postings_payload,
        }
        index_bytes = json.dumps(
            index_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        raw_instance_count = sum(len(group.instances) for group in self._groups)
        quality_instance_count = sum(len(group.quality_instances) for group in self._groups)
        quality_suppressed_nms_count = sum(
            group.quality_suppressed_nms_count for group in self._groups
        )
        quality_suppressed_containment_count = sum(
            group.quality_suppressed_containment_count for group in self._groups
        )
        self.index_manifest: dict[str, Any] = {
            "index_version": LOOKUP_INDEX_VERSION,
            "match": match_mode,
            "alias_map_enabled": bool(self._alias_map),
            "alias_map_version": self._alias_map_version,
            "manifest_sha256": self.artifact.manifest_sha256,
            "objects_sha256": self.artifact.objects_sha256,
            "frame_status_sha256": self.artifact.frame_status_sha256,
            "failure_ledger_sha256": self.artifact.failure_ledger_sha256,
            "identity_hash": self.artifact.manifest.get("identity_hash"),
            "record_count": len(self.artifact.records),
            "group_count": len(self._groups),
            "unique_label_count": len(self._postings),
            "quality_postprocess": self.quality_config.as_dict(),
            "raw_instance_count": raw_instance_count,
            "quality_instance_count": quality_instance_count,
            "quality_suppressed_nms_count": quality_suppressed_nms_count,
            "quality_suppressed_containment_count": quality_suppressed_containment_count,
            "postings_sha256": _sha256_bytes(postings_bytes),
            "index_sha256": _sha256_bytes(index_bytes),
        }

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: Path,
        *,
        allow_engineering_proxy: bool = False,
        expected_frame_uids: set[str] | None = None,
        alias_map: Mapping[str, str] | None = None,
        alias_map_version: str | None = None,
        quality_config: ObjectQualityPostprocessConfig | None = None,
    ) -> RfdetrObjectSidecarAdapter:
        return cls(
            load_rfdetr_object_sidecar(
                artifact_dir,
                allow_engineering_proxy=allow_engineering_proxy,
                expected_frame_uids=expected_frame_uids,
                quality_config=quality_config,
            ),
            alias_map=alias_map,
            alias_map_version=alias_map_version,
        )

    @property
    def frame_uids(self) -> set[str]:
        return set(self.artifact.frame_status)

    @property
    def index_version(self) -> str:
        return LOOKUP_INDEX_VERSION

    def object_aliases(self) -> dict[str, Any]:
        """Return the lightweight query-time alias catalog for the UI."""

        return {
            "status": "ready",
            "version": self._alias_map_version,
            "match": (
                "exact_normalized_label_raw_with_aliases"
                if self._alias_map
                else "exact_normalized_label_raw"
            ),
            "aliases": [
                {"alias": alias, "label": target}
                for alias, target in sorted(self._alias_map.items())
            ],
            "labels": sorted(self._postings),
        }

    def validate_frame_uid_coverage(self, expected_frame_uids: set[str]) -> None:
        expected = {str(frame_uid) for frame_uid in expected_frame_uids}
        actual = self.frame_uids
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise RfdetrObjectSidecarArtifactError(
                "canonical frame_uid coverage mismatch: "
                f"missing_sample={missing}, extra_sample={extra}"
            )

    @property
    def provider(self) -> str:
        return str(self.artifact.manifest["provenance"]["label_source"])

    @property
    def revision(self) -> str:
        model = self.artifact.manifest["model_contract"]
        return f"{model['model_id']}@{model['model_weights_sha256']}"

    @property
    def execution_status(self) -> str:
        return "ENGINEERING_PROXY"

    @property
    def quality_status(self) -> str:
        return str(self.artifact.manifest.get("quality_status", "UNVALIDATED"))

    @property
    def dataset_manifest_hash(self) -> str | None:
        return None

    @property
    def artifact_hash(self) -> str | None:
        return self.artifact.objects_sha256

    def channel_contract(self) -> ChannelContract:
        return ChannelContract(
            channel="object",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.execution_status,
            quality_status=self.quality_status,
            dataset_manifest_hash=self.dataset_manifest_hash,
            artifact_hash=self.artifact_hash,
            status="ready",
            evidence={
                "manifest_sha256": self.artifact.manifest_sha256,
                "frame_status_sha256": self.artifact.frame_status_sha256,
                "failure_ledger_sha256": self.artifact.failure_ledger_sha256,
                "identity_hash": self.artifact.manifest.get("identity_hash"),
                "index_manifest": dict(self.index_manifest),
            },
        )

    def _resolve_query_label(self, parsed_query: ParsedObjectQuery) -> tuple[str, bool]:
        query_label = parsed_query.label
        if query_label in self._postings:
            return query_label, False
        matched_label = self._alias_map.get(query_label, "")
        return matched_label, bool(matched_label)

    def _qualified_frame_uids(self, parsed_query: ParsedObjectQuery) -> set[str]:
        matched_label, _ = self._resolve_query_label(parsed_query)
        return {
            self._groups[group_index].frame_uid
            for group_index in self._postings.get(matched_label, [])
            if len(self._groups[group_index].quality_instances) >= parsed_query.min_instances
        }

    def search(self, text: str, top_k: int = 100) -> list[ChannelHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        clauses = parse_object_query_list(text)
        if not clauses:
            return []
        if len(clauses) == 1:
            return self._search_single_clause(clauses[0], top_k=top_k)

        qualified_sets = [self._qualified_frame_uids(clause) for clause in clauses]
        common_uids = set.intersection(*qualified_sets) if qualified_sets else set()
        if not common_uids:
            return []

        clause_hits = [
            {
                hit.frame_uid: hit
                for hit in self._search_single_clause(
                    clause,
                    top_k=len(common_uids),
                    allowed_frame_uids=common_uids,
                )
            }
            for clause in clauses
        ]
        combined: list[ChannelHit] = []
        for frame_uid in common_uids:
            hits = [hits_by_uid[frame_uid] for hits_by_uid in clause_hits]
            base = hits[0]
            clause_evidence = [
                dict(hit.evidence.get("channel_specific", hit.evidence)) for hit in hits
            ]
            scores = [float(hit.score) for hit in hits]
            combined_score = sum(scores) / len(scores)
            matched_labels = [str(item.get("matched_label") or "") for item in clause_evidence]
            query_labels = [str(item.get("query_label") or "") for item in clause_evidence]
            combined_specific = {
                "label": " + ".join(matched_labels),
                "label_raw": " + ".join(matched_labels),
                "label_canonical": None,
                "raw_labels": matched_labels,
                "normalized_label": " + ".join(matched_labels),
                "max_confidence": max(scores),
                "raw_max_confidence": max(
                    float(item.get("raw_max_confidence") or item.get("max_confidence") or 0.0)
                    for item in clause_evidence
                ),
                "quality_confidence": combined_score,
                "confidence": combined_score,
                "score": combined_score,
                "instance_count": None,
                "raw_instance_count": None,
                "quality_instance_count": None,
                "bbox": None,
                "raw_bbox": None,
                "position": None,
                "source_record_count": sum(
                    int(item.get("source_record_count") or 0) for item in clause_evidence
                ),
                "duplicate_group": any(
                    bool(item.get("duplicate_group")) for item in clause_evidence
                ),
                "duplicate_extra_instance_count": sum(
                    int(item.get("duplicate_extra_instance_count") or 0) for item in clause_evidence
                ),
                "object_query_mode": "all",
                "object_clause_count": len(clause_evidence),
                "object_clauses": clause_evidence,
                "query_label": " + ".join(query_labels),
                "raw_query": " + ".join(clause.raw_query for clause in clauses),
                "min_instances": None,
                "quantity_satisfied": True,
                "matched_label": " + ".join(matched_labels),
                "matched_by_alias": any(
                    bool(item.get("matched_by_alias")) for item in clause_evidence
                ),
                "alias_map_version": self._alias_map_version,
            }
            evidence = build_channel_evidence(
                channel="object",
                provider=self.provider,
                revision=self.revision,
                execution_status=self.execution_status,
                quality_status=self.quality_status,
                dataset_manifest_hash=self.dataset_manifest_hash,
                artifact_hash=self.artifact_hash,
                frame_uid=str(base.frame_uid or base.entity_id),
                video_id=base.video_id,
                video_filename=str(base.video_filename or f"{base.video_id}.mp4"),
                source_frame_idx=int(base.source_frame_idx or 0),
                timestamp_ms=base.timestamp_ms,
                score=combined_score,
                rank=1,
                channel_specific=combined_specific,
                raw_provenance=dict(base.evidence.get("raw_provenance", {})),
            )
            combined.append(
                ChannelHit(
                    entity_id=str(base.entity_id),
                    video_id=base.video_id,
                    timestamp_ms=base.timestamp_ms,
                    modality="object",
                    score=combined_score,
                    rank=1,
                    provider=self.provider,
                    evidence_text=" + ".join(matched_labels),
                    frame_uid=base.frame_uid,
                    video_filename=base.video_filename,
                    source_frame_idx=base.source_frame_idx,
                    evidence=evidence,
                )
            )
        combined.sort(
            key=lambda hit: (
                -float(hit.score),
                hit.video_id,
                int(hit.source_frame_idx or 0),
                str(hit.frame_uid or hit.entity_id),
            )
        )
        results: list[ChannelHit] = []
        for rank, hit in enumerate(combined[:top_k], start=1):
            evidence = dict(hit.evidence)
            evidence["rank"] = rank
            evidence["score"] = float(hit.score)
            evidence["channel_specific"] = dict(evidence.get("channel_specific", {}))
            evidence["channel_specific"]["rank"] = rank
            evidence["channel_specific"]["score"] = float(hit.score)
            evidence.update(evidence["channel_specific"])
            results.append(
                ChannelHit(
                    entity_id=hit.entity_id,
                    video_id=hit.video_id,
                    timestamp_ms=hit.timestamp_ms,
                    modality=hit.modality,
                    score=hit.score,
                    rank=rank,
                    provider=hit.provider,
                    evidence_text=hit.evidence_text,
                    frame_uid=hit.frame_uid,
                    video_filename=hit.video_filename,
                    source_frame_idx=hit.source_frame_idx,
                    evidence=evidence,
                )
            )
        return results

    def _search_single_clause(
        self,
        parsed_query: ParsedObjectQuery,
        *,
        top_k: int,
        allowed_frame_uids: set[str] | None = None,
    ) -> list[ChannelHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_label = parsed_query.label
        if not query_label:
            return []
        matched_label, matched_by_alias = self._resolve_query_label(parsed_query)
        candidates = self._ranked_postings.get(matched_label, ())
        if not candidates:
            return []
        scored: list[tuple[_GroupedHit, float]] = []
        for group_index in candidates:
            group = self._groups[group_index]
            if allowed_frame_uids is not None and group.frame_uid not in allowed_frame_uids:
                continue
            if len(group.quality_instances) < parsed_query.min_instances:
                continue
            scored.append((group, float(group.confidence)))
        results: list[ChannelHit] = []
        for rank, (group, score) in enumerate(scored[:top_k], start=1):
            bbox = list(group.bbox) if group.bbox is not None else None
            position = None
            if group.bbox is not None:
                x1, y1, x2, y2 = group.bbox
                status = group.frame_status or {}
                width = float(status.get("width") or 0.0)
                height = float(status.get("height") or 0.0)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                position = {
                    "center_x": center_x,
                    "center_y": center_y,
                    "normalized_center": {
                        "x": center_x / width if width > 0 else None,
                        "y": center_y / height if height > 0 else None,
                    },
                    "source": "derived_from_bbox",
                }
            channel_specific = {
                "label": group.raw_labels[0] if group.raw_labels else group.normalized_label,
                "label_raw": group.raw_labels[0] if group.raw_labels else group.normalized_label,
                "label_canonical": None,
                "raw_labels": list(group.raw_labels),
                "normalized_label": group.normalized_label,
                "max_confidence": group.confidence,
                "raw_max_confidence": group.raw_confidence,
                "quality_confidence": group.confidence,
                "confidence": group.confidence,
                "score": score,
                "instance_count": len(group.quality_instances),
                "raw_instance_count": len(group.instances),
                "quality_instance_count": len(group.quality_instances),
                "bbox": bbox,
                "raw_bbox": list(group.raw_bbox) if group.raw_bbox is not None else None,
                "position": position,
                "source_record_count": len(group.records),
                "duplicate_group": len(group.records) > 1,
                "duplicate_extra_instance_count": len(group.instances) - len(group.records),
                "instances": [dict(instance) for instance in group.instances],
                "quality_instances": [dict(instance) for instance in group.quality_instances],
                "quality_postprocess": {
                    **self.quality_config.as_dict(),
                    "raw_instance_count": len(group.instances),
                    "quality_instance_count": len(group.quality_instances),
                    "suppressed_nms_count": group.quality_suppressed_nms_count,
                    "suppressed_containment_count": group.quality_suppressed_containment_count,
                    "suppressed_count": (
                        group.quality_suppressed_nms_count
                        + group.quality_suppressed_containment_count
                    ),
                },
                "source_records": [dict(record) for record in group.records],
                "source_shard_ids": list(group.source_shard_ids),
                "frame_status": dict(group.frame_status or {}),
                "model_id": group.model_id,
                "model_weights_sha256": group.model_weights_sha256,
                "revision": group.revision,
                "provider": group.provider,
                "label_source": group.label_source,
                "provider_execution": group.provider_execution,
                "keyframe_paths": list(group.keyframe_paths),
                "query_label": query_label,
                "raw_query": parsed_query.raw_query,
                "min_instances": parsed_query.min_instances,
                "quantity_satisfied": len(group.quality_instances) >= parsed_query.min_instances,
                "matched_label": matched_label,
                "matched_by_alias": matched_by_alias,
                "alias_map_version": self._alias_map_version,
            }
            results.append(
                ChannelHit(
                    entity_id=group.frame_uid,
                    video_id=group.video_id,
                    timestamp_ms=group.timestamp_ms,
                    modality="object",
                    score=float(score),
                    rank=rank,
                    provider=self.provider,
                    evidence_text=group.raw_labels[0]
                    if group.raw_labels
                    else group.normalized_label,
                    frame_uid=group.frame_uid,
                    video_filename=group.video_filename,
                    source_frame_idx=group.source_frame_idx,
                    evidence=build_channel_evidence(
                        channel="object",
                        provider=self.provider,
                        revision=self.revision,
                        execution_status=self.execution_status,
                        quality_status=self.quality_status,
                        dataset_manifest_hash=self.dataset_manifest_hash,
                        artifact_hash=self.artifact_hash,
                        frame_uid=group.frame_uid,
                        video_id=group.video_id,
                        video_filename=group.video_filename,
                        source_frame_idx=group.source_frame_idx,
                        timestamp_ms=group.timestamp_ms,
                        score=float(score),
                        rank=rank,
                        channel_specific=channel_specific,
                        raw_provenance={
                            "manifest_sha256": self.artifact.manifest_sha256,
                            "objects_sha256": self.artifact.objects_sha256,
                            "frame_status_sha256": self.artifact.frame_status_sha256,
                            "failure_ledger_sha256": self.artifact.failure_ledger_sha256,
                            "identity_hash": self.artifact.manifest.get("identity_hash"),
                            "source_execution": self.artifact.manifest.get("provenance", {}).get(
                                "source_execution"
                            ),
                            "artifact_manifest_provenance": dict(
                                self.artifact.manifest.get("provenance", {})
                            ),
                            "model_contract": dict(
                                self.artifact.manifest.get("model_contract", {})
                            ),
                            "index_manifest": dict(self.index_manifest),
                        },
                    ),
                )
            )
        return results
