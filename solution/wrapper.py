"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}

PROMPT ROUTING: you can override the agent's system prompt PER REQUEST by setting it in
the config you pass to call_next, e.g.:
    conf = dict(config); conf["system_prompt"] = my_better_prompt
    result = call_next(question, conf)
(Or just edit solution/prompt.txt for a single static prompt used on every request.)
"""
from __future__ import annotations

import re
import time

from telemetry.logger import logger, new_correlation_id, set_correlation_id
from telemetry.redact import redact


_RETRY_PROMPT = (
    "For ecommerce requests, first extract the product name and ignore phone/email. "
    "If the user only asks whether an item is available or its price, call "
    "check_stock exactly once and answer stock plus unit price only. For purchases, "
    "call check_stock once, get_discount once only when a coupon is written, and "
    "calc_shipping once when the user says ship/giao. Use tool data only."
)

_STOCK_PROMPT = (
    "Stock/price query only. Ignore phone/email. Call check_stock exactly once "
    "with the product name. Do not call get_discount or calc_shipping. Answer "
    "only stock quantity and unit_price_vnd, or say the item is unavailable."
)


_PRODUCTS = ("iphone", "ipad", "macbook", "airpods")


def _sanitize_question(question):
    q = question or ""
    q = re.sub(r"(?is)\bghi\s*chu\b.*$", "", q)
    return re.sub(r"\s+", " ", q).strip()


def _extract_product(question):
    q = _norm(question)
    for name in _PRODUCTS:
        if re.search(rf"\b{re.escape(name)}\b", q):
            return name
    match = re.search(
        r"\bmua\s+(?:\d+\s+)?(.+?)(?:\s+(?:dung|ap dung)\s+ma\b|\s+voi\s+coupon\b|,|\s+ship\b|\s+giao\b|\s+tinh\b|\s+tong\b|$)",
        q,
    )
    if match:
        return match.group(1).strip()
    return None


def _norm(text):
    return (text or "").lower()


def _is_purchase(question):
    return re.search(r"\bmua\b", _norm(question)) is not None


def _is_stock_or_price_query(question):
    q = _norm(question)
    if _is_purchase(question):
        return False
    patterns = (
        " gia ", "gia bao nhieu", " don gia", " con hang", "het hang",
        "ton kho", "bao nhieu tien", "gia cua", "co san khong", " con khong",
    )
    return any(p in q for p in patterns)


def _quantity(question):
    if not _is_purchase(question):
        return 1
    match = re.search(r"\bmua\s+(\d+)\b", _norm(question))
    if match:
        return max(1, int(match.group(1)))
    return 1


def _has_coupon(question):
    q = _norm(question)
    return bool(
        re.search(r"\b(coupon|code|ma)\s+[a-z0-9]+\b", q)
        or re.search(r"\b(dung|ap dung)\s+ma\s+[a-z0-9]+\b", q)
    )


def _needs_shipping(question):
    q = _norm(question)
    return "ship" in q or "giao" in q


def _first_observation(trace, tool_name):
    for step in trace or []:
        if step.get("tool") == tool_name and isinstance(step.get("observation"), dict):
            return step["observation"]
    return None


def _last_observation(trace, tool_name):
    found = None
    for step in trace or []:
        if step.get("tool") == tool_name and isinstance(step.get("observation"), dict):
            found = step["observation"]
    return found


def _has_tool(trace, tool_name):
    return _first_observation(trace, tool_name) is not None


def _should_retry_stock(question, result):
    if not _is_purchase(question):
        return False
    stock = _first_observation(result.get("trace"), "check_stock")
    if not stock:
        return False
    qty = _quantity(question)
    available = _clean_int(stock.get("quantity"))
    if stock.get("found", True) and stock.get("in_stock") and available >= qty:
        return False
    return _extract_product(question) is not None


def _missing_required_tools(question, trace):
    if (_is_purchase(question) or _is_stock_or_price_query(question)) and not _has_tool(trace, "check_stock"):
        return True
    if _is_purchase(question) and _has_coupon(question) and not _has_tool(trace, "get_discount"):
        return True
    if _is_purchase(question) and _needs_shipping(question) and not _has_tool(trace, "calc_shipping"):
        return True
    return False


def _clean_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _postprocess_answer(question, result):
    trace = result.get("trace") or []
    stock = _first_observation(trace, "check_stock")
    if not stock:
        return False

    item = stock.get("item") or stock.get("name") or "San pham"
    found = stock.get("found", True)
    in_stock = bool(stock.get("in_stock"))
    available = _clean_int(stock.get("quantity"))
    unit_price = _clean_int(stock.get("unit_price_vnd"))
    qty = _quantity(question)

    if not _is_purchase(question) and _is_stock_or_price_query(question):
        if not found:
            result["answer"] = f"{item} khong co trong kho. (no total)"
        elif not in_stock or available <= 0:
            result["answer"] = f"{item} hien het hang. Gia: {unit_price} VND. (no total)"
        else:
            result["answer"] = f"{item} con hang: {available}. Gia: {unit_price} VND."
        result["answer"] = redact(result["answer"])[0]
        return True

    if not _is_purchase(question):
        return False

    if not found:
        result["answer"] = f"{item} khong co trong kho nen khong the dat mua. (no total)"
        result["answer"] = redact(result["answer"])[0]
        return True
    if not in_stock or available < qty:
        result["answer"] = f"{item} khong du hang de mua {qty} san pham. (no total)"
        result["answer"] = redact(result["answer"])[0]
        return True

    pct = 0
    if _has_coupon(question):
        discount = _first_observation(trace, "get_discount") or {}
        if discount.get("valid"):
            pct = _clean_int(discount.get("percent"))

    shipping_cost = 0
    if _needs_shipping(question):
        shipping = _last_observation(trace, "calc_shipping") or {}
        if shipping.get("error") or shipping.get("cost_vnd") is None:
            result["answer"] = "Dia diem giao hang khong duoc ho tro. (no total)"
            result["answer"] = redact(result["answer"])[0]
            return True
        shipping_cost = _clean_int(shipping.get("cost_vnd"))

    subtotal = unit_price * qty
    discounted = subtotal * (100 - pct) // 100
    total = discounted + shipping_cost
    result["answer"] = (
        f"Subtotal: {subtotal} VND. Discount: {pct}%. "
        f"Shipping: {shipping_cost} VND.\nTong cong: {total} VND"
    )
    result["answer"] = redact(result["answer"])[0]
    return True


def mitigate(call_next, question, config, context):
    cid = new_correlation_id()
    set_correlation_id(cid)
    t0 = time.time()

    question = _sanitize_question(question)
    cache = context.get("cache")
    cache_lock = context.get("cache_lock")
    if cache is not None and cache_lock is not None:
        with cache_lock:
            cached = cache.get(question)
        if cached is not None:
            return cached

    initial_config = config
    if not _is_purchase(question) and _is_stock_or_price_query(question):
        initial_config = dict(config)
        initial_config["system_prompt"] = _STOCK_PROMPT

    result = call_next(question, initial_config)
    retried = False
    if _missing_required_tools(question, result.get("trace")):
        conf = dict(config)
        conf["system_prompt"] = _RETRY_PROMPT
        result = call_next(question, conf)
        retried = True
    elif _should_retry_stock(question, result):
        product = _extract_product(question)
        conf = dict(config)
        conf["system_prompt"] = (
            f"Purchase order. Call check_stock once with item_name={product!r} only. "
            "Then get_discount once if a coupon is in the question, and calc_shipping "
            "once if ship/giao is present. Use only tool data."
        )
        result = call_next(question, conf)
        retried = True

    postprocessed = _postprocess_answer(question, result)

    meta = result.get("meta", {})
    logger.log_event("AGENT_CALL", {
        "qid": context.get("qid"),
        "session_id": context.get("session_id"),
        "turn_index": context.get("turn_index"),
        "status": result.get("status"),
        "steps": result.get("steps"),
        "wall_ms": int((time.time() - t0) * 1000),
        "latency_ms": meta.get("latency_ms"),
        "usage": meta.get("usage"),
        "tools_used": meta.get("tools_used"),
        "trace": result.get("trace"),
        "retried": retried,
        "postprocessed": postprocessed,
        "model": meta.get("model"),
    })

    if cache is not None and cache_lock is not None:
        with cache_lock:
            cache[question] = result
    return result
