"""The teacher: a large LLM labels ads and explains its decision.

Usage (run from the project root):
    python -m aldau.teacher label --limit 10               # label up to 10 ads (asks y/n first)
    python -m aldau.teacher label --ids a1b2c3,d4e5f6 --show   # chosen ads, print full answers
    python -m aldau.teacher label --sample --limit 10      # 10 random ads (fixed seed)
    python -m aldau.teacher compare                        # teacher vs human labels in raw
    python -m aldau.teacher compare --gold                 # teacher vs the gold set
    python -m aldau.teacher versions --old v1              # old vs current prompt, side by side

How it saves money and survives crashes:
- several ads go into one request (a "batch", 10 by default);
- every answer is saved per ad in data/cache/teacher.jsonl as soon as it arrives;
- before a run, ads already in the cache are skipped. After a crash or a hit daily
  limit, just run the same command again — it continues where it stopped.

The cache key is a hash of: provider + model + PROMPT_VERSION + the ad's own fields.
Change the prompt -> bump PROMPT_VERSION -> old answers are not reused by mistake.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from aldau import llm
from aldau.collect import GOLD_PATH, RAW_PATH, read_ads
from aldau.features import SIGNAL_DESCRIPTIONS_RU, PriceReference, find_signals
from aldau.schema import Ad, TeacherLabel

# v1 — first prompt; risk scores came out bimodal (0.05 or >0.9).
# v2 — explicit risk scale with examples for every range (calibration).
PROMPT_VERSION = "v2"
CACHE_PATH = Path("data/cache/teacher.jsonl")
LABELS_PATH = Path("data/labeled/teacher_labels.jsonl")
REPORT_PATH = Path("reports/teacher_check.md")
BATCH_SIZE = 10
DEFAULT_LIMIT = 10  # cost guard: a small default, raise it on purpose
PAUSE_SECONDS = 5.0  # between requests, to stay under the free tier's per-minute limit


# --- answer shape ------------------------------------------------------------------


class TeacherItem(TeacherLabel):
    """One ad's answer inside a batch: the label plus the ad id it belongs to."""

    id: str


class TeacherBatch(BaseModel):
    """The whole answer to one request. Also sent to the model as the required JSON schema."""

    results: list[TeacherItem]


# --- prompt ------------------------------------------------------------------------

INSTRUCTIONS = """\
Ты — эксперт по мошенничеству на казахстанских площадках объявлений (OLX.kz, Kolesa.kz,
Krisha.kz). Тебе дают объявления на русском или казахском. Для КАЖДОГО объявления оцени,
мошенническое оно или обычное, и объясни почему.

Типичные схемы мошенничества в Казахстане:
- предоплата, задаток или «бронь» до того, как покупатель увидел квартиру, машину или товар;
- аренда: «хозяин в другом городе / за границей», ключи «передаст знакомый после оплаты»,
  показ только после перевода, «желающих много — кто первый оплатит»;
- посуточная аренда и базы отдыха: полная оплата вперёд на карту частному лицу, адрес
  «после перевода»;
- цена сильно ниже рынка вместе со срочностью («уезжаю», «срочно нужны деньги»);
- «доставка OLX» или «безопасная сделка» по ссылке, где просят ввести данные карты;
- товар или машина «из Кореи / ОАЭ / Грузии», «уже на таможне — помогите закрыть платёж»,
  100 % предоплата за пригон;
- новые дорогие телефоны в 3–5 раз дешевле магазина, «акция», «количество ограничено»;
- продавец обещает прислать фото удостоверения или документов, чтобы вызвать доверие;
- общение только в мессенджере, на звонки не отвечает;
- «квартира по госпрограмме без очереди», «есть знакомые в акимате» — за взнос;
- работа с оплатой «за обучение», «за оформление», «за форму».

Что НЕ является мошенничеством само по себе:
- депозит при долгосрочной аренде (обычная практика), особенно «при заселении»;
- бронь в гостинице, хостеле или у официальной компании посуточной аренды;
- автосалоны, дилеры, магазины, агентства с рассрочкой, гарантией, чеками;
- слово «срочно», торг, просьба писать в WhatsApp, низкая цена за старую или сломанную вещь;
- безграмотный текст или смесь русского и казахского.
Решает сочетание признаков: схема обмана — это когда покупателя просят рискнуть деньгами,
не дав ему проверить товар и продавца.

Поля каждого объявления: id, category, city, price_kzt, price_ratio (цена, делённая на
медианную цену в этой категории по нашей небольшой выборке: 0.3 = на 70 % дешевле обычного;
null = неизвестно; выборка маленькая, относись к этому числу осторожно), title, text и
regex_hints — признаки, найденные регулярными выражениями. Подсказки могут ошибаться
(например, «депозит» в честной аренде): проверяй их по тексту и не повторяй вслепую.
Личные данные в тексте заменены метками [PHONE], [LINK], [EMAIL], [USERNAME], [NAME].

Верни JSON {"results": [...]} — ровно один элемент на каждое объявление, с тем же id:
- label: "scam" или "legit";
- risk_score: число от 0 до 1 — риск мошенничества. Пользуйся всей шкалой, а не только
  крайними значениями; выбирай число внутри диапазона по силе признаков:
  * 0.00–0.15 — признаков нет. Пример: «Продам стиральную машину LG, 6 кг, рабочая,
    самовывоз, звоните» → 0.05.
  * 0.15–0.40 — один слабый признак или неоднозначность, схемы нет. Примеры: «СРОЧНО продаю,
    переезд, торг» → 0.2; «звонки не отвечаю, пишите в WhatsApp» → 0.25; цена заметно ниже
    рынка без объяснения → 0.3; депозит при посуточной аренде «при заселении» → 0.15.
  * 0.40–0.70 — несколько настораживающих признаков, но узнаваемой схемы нет. Примеры:
    «продавец в другом городе, отправлю почтой» без требования предоплаты → 0.5;
    очень низкая цена + срочность + только мессенджер → 0.6.
  * 0.70–1.00 — узнаваемая схема обмана. Примеры: «ключи после оплаты первого месяца на
    карту, хозяин в другом городе» → 0.95; «оплатите по ссылке OLX-доставки, введите данные
    карты» → 0.98; «бронь 30 000, желающих много» до просмотра → 0.85.
  label = "scam", если risk_score >= 0.5, иначе "legit";
- signals: короткие названия найденных признаков латиницей. Используй по возможности эти:
  {signal_names}, price_too_low, insider_promise, fake_job;
  для честного объявления — пустой список или нейтральные признаки не указывай вовсе;
- explanation_ru: 1–3 предложения по-русски, со ссылкой на конкретные слова из объявления.
  Не выдумывай фактов, которых нет в тексте.

Объявления:
"""


def ad_for_prompt(ad: Ad, prices: PriceReference) -> dict:
    """What the teacher sees about one ad.

    Deliberately NOT included: our human label and the origin (real / case_based).
    If the teacher saw them, it would copy the answer instead of judging the text.
    """
    ratio = prices.ratio(ad)
    return {
        "id": ad.id,
        "category": ad.category,
        "city": ad.city,
        "price_kzt": ad.price_kzt,
        "price_ratio": round(ratio, 2) if ratio is not None else None,
        "title": ad.title,
        "text": ad.text,
        "regex_hints": [
            SIGNAL_DESCRIPTIONS_RU[name]
            for name in find_signals(ad.title + "\n" + ad.text)
            if name not in ("has_phone", "has_link")  # masked contacts are not hints
        ],
    }


def build_prompt(ads: list[Ad], prices: PriceReference) -> str:
    signal_names = ", ".join(name for name in SIGNAL_DESCRIPTIONS_RU if name != "has_phone")
    items = [ad_for_prompt(ad, prices) for ad in ads]
    instructions = INSTRUCTIONS.replace("{signal_names}", signal_names)
    return instructions + json.dumps(items, ensure_ascii=False, indent=1)


# --- cache -------------------------------------------------------------------------


def cache_key(ad: Ad, model: str, version: str | None = None) -> str:
    """Hash of everything that decides the answer.

    `version` defaults to the current PROMPT_VERSION; pass an old one to read old answers.
    price_ratio is left out on purpose: it changes every time new ads are imported
    (the medians move), and re-paying for every ad after each import is not worth
    the small difference.
    """
    version = version or PROMPT_VERSION
    parts = [model, version, ad.title, ad.text, str(ad.price_kzt), ad.category, str(ad.city)]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


def read_cache(path: Path = CACHE_PATH) -> dict[str, dict]:
    """key -> cached record. A broken line (e.g. the PC died mid-write) is skipped."""
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                records[record["key"]] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def append_cache(path: Path, key: str, ad: Ad, model: str, label: TeacherLabel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "ad_id": ad.id,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label.model_dump(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cached_labels(
    ads: list[Ad], model: str, path: Path = CACHE_PATH, version: str | None = None
) -> dict[str, TeacherLabel]:
    """ad id -> teacher label, for the ads this model + prompt version already labelled."""
    cache = read_cache(path)
    labels = {}
    for ad in ads:
        record = cache.get(cache_key(ad, model, version))
        if record:
            labels[ad.id] = TeacherLabel.model_validate(record["label"])
    return labels


# --- labelling ---------------------------------------------------------------------


@dataclass
class RunReport:
    requests: int = 0
    labelled: list[str] = field(default_factory=list)  # ad ids saved in this run
    problems: list[str] = field(default_factory=list)  # human-readable
    stopped_early: bool = False


def parse_batch(answer: dict, batch: list[Ad]) -> tuple[dict[str, TeacherLabel], list[str]]:
    """Check the model's answer item by item.

    Returns the valid labels (ad id -> label) and a list of problems. An item with a
    wrong id, a score outside 0..1 or a missing field is dropped and NOT cached,
    so that ad is simply asked again next run.
    """
    wanted = {ad.id for ad in batch}
    labels: dict[str, TeacherLabel] = {}
    problems = []
    items = answer.get("results") if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return {}, ["в ответе нет списка results"]
    for item in items:
        try:
            parsed = TeacherItem.model_validate(item)
        except ValidationError as error:
            problems.append(f"элемент не прошёл проверку схемы: {error.errors()[0]['msg']}")
            continue
        if parsed.id not in wanted:
            problems.append(f"чужой id в ответе: {parsed.id}")
            continue
        labels[parsed.id] = TeacherLabel.model_validate(parsed.model_dump(exclude={"id"}))
    missing = wanted - set(labels)
    if missing:
        problems.append(f"нет ответа для {len(missing)} объявл.: {', '.join(sorted(missing))}")
    return labels, problems


def label_ads(
    ads: list[Ad],
    prices: PriceReference,
    model: str,
    batch_size: int = BATCH_SIZE,
    pause: float = PAUSE_SECONDS,
    cache_path: Path = CACHE_PATH,
    ask: Callable[[str, type[BaseModel]], dict] = llm.complete_json,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> RunReport:
    """Label the ads that are not cached yet, batch by batch.

    A permanent API error (bad key, daily limit, retries exhausted) stops the run:
    everything labelled so far is already in the cache.
    `ask` is llm.complete_json in real use; tests pass a fake.
    """
    report = RunReport()
    cache = read_cache(cache_path)
    todo = [ad for ad in ads if cache_key(ad, model) not in cache]
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]

    for number, batch in enumerate(batches, start=1):
        if number > 1:
            sleep(pause)
        log(f"Запрос {number}/{len(batches)}: {len(batch)} объявл.")
        try:
            answer = ask(build_prompt(batch, prices), TeacherBatch)
        except llm.LLMError as error:
            report.problems.append(f"запрос {number}: {error}")
            report.stopped_early = True
            log(f"  Остановка: {error}. Уже размеченное сохранено в кеше.")
            break
        report.requests += 1
        labels, problems = parse_batch(answer, batch)
        report.problems += [f"запрос {number}: {p}" for p in problems]
        for ad in batch:
            if ad.id in labels:
                append_cache(cache_path, cache_key(ad, model), ad, model, labels[ad.id])
                report.labelled.append(ad.id)
    return report


def export_labels(
    ads: list[Ad], model: str, cache_path: Path = CACHE_PATH, out_path: Path = LABELS_PATH
) -> int:
    """Write data/labeled/teacher_labels.jsonl: one line per ad the teacher has labelled."""
    labels = cached_labels(ads, model, cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ad in ads:
            if ad.id in labels:
                row = {"id": ad.id, "model": model, "prompt_version": PROMPT_VERSION}
                row.update(labels[ad.id].model_dump())
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(labels)


# --- comparing with human labels --------------------------------------------------


def classification_metrics(pairs: list[tuple[str, TeacherLabel]]) -> dict:
    """Metrics for (human label, teacher label) pairs; 'scam' is the positive class.

    precision = of the ads the teacher called scam, how many really are scam;
    recall    = of the real scam ads, how many the teacher caught;
    F1        = the harmonic mean of the two (low if either one is low).
    """
    tp = sum(h == "scam" and t.label == "scam" for h, t in pairs)
    fp = sum(h == "legit" and t.label == "scam" for h, t in pairs)
    fn = sum(h == "scam" and t.label == "legit" for h, t in pairs)
    tn = sum(h == "legit" and t.label == "legit" for h, t in pairs)
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan
    return {
        "n": len(pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(pairs) if pairs else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fmt(value: float) -> str:
    return "—" if value != value else f"{value:.2f}"  # NaN != NaN


def compare_report(ads: list[Ad], labels: dict[str, TeacherLabel], model: str, source: str) -> str:
    """Markdown report: overall and separately for real / case_based (no ad texts inside)."""
    from aldau.eda import roc_auc  # imported here to keep the module import light

    judged = [ad for ad in ads if ad.label in ("scam", "legit") and ad.id in labels]
    slices = [("все", judged)] + [
        (origin, [ad for ad in judged if ad.origin == origin]) for origin in ("real", "case_based")
    ]
    lines = [
        "# Проверка учителя",
        "",
        f"Модель: `{model}`, версия промпта: `{PROMPT_VERSION}`. Эталон: {source}.",
        f"Объявлений с человеческой меткой и ответом учителя: {len(judged)}.",
        "",
        "| срез | n | accuracy | precision | recall | F1 | ROC-AUC | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, group in slices:
        if not group:
            lines.append(f"| {name} | 0 | — | — | — | — | — | | | | |")
            continue
        m = classification_metrics([(ad.label, labels[ad.id]) for ad in group])
        auc = roc_auc(
            [labels[ad.id].risk_score for ad in group], [ad.label == "scam" for ad in group]
        )
        lines.append(
            f"| {name} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m['precision'])} "
            f"| {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(auc)} "
            f"| {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    lines += [
        "",
        "ROC-AUC считается по risk_score. TP — scam, и учитель сказал scam; "
        "FP — честное, а учитель сказал scam; "
        "FN — пропущенный scam; TN — честное, и учитель сказал legit.",
        "Срез `case_based` содержит только scam (переписанные кейсы), поэтому precision там "
        "не показателен; важнее recall.",
    ]
    return "\n".join(lines + recall_text_section(judged, labels)) + "\n"


def recall_text_section(judged: list[Ad], labels: dict[str, TeacherLabel]) -> list[str]:
    """Recall with and without the scam ads marked "по тексту не распознать" (key t in gold).

    A model that only reads the text cannot catch those, so recall is shown both ways:
    with them (the honest number for users) and without them (what the text allows).
    """
    scam = [ad for ad in judged if ad.label == "scam"]
    visible = [ad for ad in scam if not ad.not_from_text]

    def recall(group: list[Ad]) -> str:
        caught = sum(labels[ad.id].label == "scam" for ad in group)
        return f"{fmt(caught / len(group)) if group else '—'} ({caught} из {len(group)})"

    return [
        "",
        "## Recall и объявления «по тексту не распознать»",
        "",
        f"- recall по всем scam: {recall(scam)}",
        f"- recall без помеченных «по тексту не распознать»: {recall(visible)}",
        f"- помечено таких scam: {len(scam) - len(visible)}",
    ]


# --- command line -------------------------------------------------------------------


def select_ads(ads: list[Ad], ids: str | None, sample: bool, limit: int) -> list[Ad]:
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {ad.id: ad for ad in ads}
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            raise SystemExit(f"Нет объявлений с id: {', '.join(unknown)}")
        chosen = [by_id[i] for i in wanted]
    elif sample:
        chosen = ads[:]
        random.Random(42).shuffle(chosen)  # fixed seed: the same sample every time
    else:
        chosen = ads
    return chosen[:limit]


def show_answer(ad: Ad, label: TeacherLabel) -> None:
    print("\n" + "=" * 78)
    print(f"[{ad.id}] {ad.source} | {ad.category} | {ad.city or '-'} | {ad.price_kzt} тг")
    print(f"Наша метка: {ad.label} ({ad.origin})")
    print(f"{ad.title}\n{ad.text[:600]}{' …' if len(ad.text) > 600 else ''}")
    print("-" * 78)
    print(json.dumps(label.model_dump(), ensure_ascii=False, indent=2))


def cmd_label(args: argparse.Namespace) -> None:
    model = llm.model_id()
    all_ads = read_ads(RAW_PATH)
    prices = PriceReference(min_count=3).fit(all_ads)
    chosen = select_ads(all_ads, args.ids, args.sample, args.limit)
    cache = read_cache(CACHE_PATH)
    todo = [ad for ad in chosen if cache_key(ad, model) not in cache]
    requests = math.ceil(len(todo) / args.batch_size)

    print(f"Модель: {model}, промпт {PROMPT_VERSION}")
    print(f"Выбрано объявлений: {len(chosen)}, уже в кеше: {len(chosen) - len(todo)}")
    print(f"К разметке: {len(todo)} -> запросов к API: {requests} (по {args.batch_size} в запросе)")
    if requests and not args.yes:
        if input("Продолжить? (y/n) > ").strip().lower() != "y":
            print("Отменено, запросов не было.")
            return
    report = label_ads(chosen, prices, model, batch_size=args.batch_size, pause=args.pause)

    print(f"\nЗапросов выполнено: {report.requests}, размечено сейчас: {len(report.labelled)}")
    for problem in report.problems:
        print(f"  ! {problem}")
    print(f"Всего с меткой учителя: {export_labels(all_ads, model)} -> {LABELS_PATH}")
    if args.show:
        labels = cached_labels(chosen, model)
        for ad in chosen:
            if ad.id in labels:
                show_answer(ad, labels[ad.id])


def cmd_compare(args: argparse.Namespace) -> None:
    model = llm.model_id()
    ads = read_ads(GOLD_PATH if args.gold else RAW_PATH)
    source = "gold-набор (ручная разметка)" if args.gold else "метки в data/raw (не gold)"
    labels = cached_labels(ads, model)
    report = compare_report(ads, labels, model, source)
    print(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Сохранено в {REPORT_PATH}")
    disagreements = [
        ad
        for ad in ads
        if ad.id in labels and ad.label in ("scam", "legit") and labels[ad.id].label != ad.label
    ]
    if disagreements:
        print(f"\nРасхождения с человеком: {len(disagreements)}")
        for ad in disagreements:
            t = labels[ad.id]
            print(
                f"  [{ad.id}] человек {ad.label}, учитель {t.label} "
                f"({t.risk_score:.2f}) — {ad.title[:50]}"
            )


def compare_versions(
    ads: list[Ad], model: str, old: str, new: str, path: Path = CACHE_PATH
) -> list[str]:
    """Side-by-side table of two prompt versions for the ads both have labelled."""
    old_labels = cached_labels(ads, model, path, old)
    new_labels = cached_labels(ads, model, path, new)
    both = [ad for ad in ads if ad.id in old_labels and ad.id in new_labels]
    lines = [
        f"| id | наша метка | {old} | {new} | объявление |",
        "|---|---|---|---|---|",
    ]
    for ad in both:
        a, b = old_labels[ad.id], new_labels[ad.id]
        mark = " ⚠️" if a.label != b.label else ""
        lines.append(
            f"| {ad.id} | {ad.label} | {a.label} {a.risk_score:.2f} | {b.label} {b.risk_score:.2f}"
            f"{mark} | {ad.title[:45]} |"
        )
    for version, labels in ((old, old_labels), (new, new_labels)):
        scores = [labels[ad.id].risk_score for ad in both]
        if scores:
            middle = sum(0.15 < s < 0.7 for s in scores)
            lines.append(
                f"\n{version}: разных значений риска {len(set(scores))}, "
                f"в диапазоне 0.15–0.70: {middle} из {len(scores)}"
            )
    return lines


def cmd_versions(args: argparse.Namespace) -> None:
    model = llm.model_id()
    ads = read_ads(RAW_PATH)
    if args.ids:
        ads = select_ads(ads, args.ids, sample=False, limit=len(ads))
    print("\n".join(compare_versions(ads, model, args.old, args.new or PROMPT_VERSION)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher LLM: label ads and check the labels.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_label = sub.add_parser("label", help="label ads with the teacher (calls the API)")
    p_label.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max ads (cost guard)")
    p_label.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_label.add_argument(
        "--pause", type=float, default=PAUSE_SECONDS, help="seconds between requests"
    )
    p_label.add_argument("--ids", help="comma-separated ad ids to label")
    p_label.add_argument("--sample", action="store_true", help="random ads (fixed seed)")
    p_label.add_argument("--show", action="store_true", help="print the full answers")
    p_label.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p_label.set_defaults(func=cmd_label)

    p_compare = sub.add_parser("compare", help="teacher vs human labels (no API calls)")
    p_compare.add_argument("--gold", action="store_true", help="use the gold set")
    p_compare.set_defaults(func=cmd_compare)

    p_versions = sub.add_parser("versions", help="compare two prompt versions (no API calls)")
    p_versions.add_argument("--old", required=True, help="old prompt version, e.g. v1")
    p_versions.add_argument("--new", help="new prompt version (default: current)")
    p_versions.add_argument("--ids", help="comma-separated ad ids")
    p_versions.set_defaults(func=cmd_versions)

    args = parser.parse_args()
    try:
        args.func(args)
    except llm.LLMError as error:
        raise SystemExit(f"Ошибка настроек LLM: {error}") from error


if __name__ == "__main__":
    main()
