"""
Runnable Python version of homework2_part1_submission.ipynb

Why this file exists:
- nbconvert generated .py contains top-level `await`, which cannot run via `python file.py`.
- This runner wraps all async calls in `asyncio.run(...)`.

Dependencies (install manually; this script will NOT install packages):
  pip install gdown "markitdown[pdf]" langchain-google-genai langchain-mcp-adapters langchain-core httpx

Usage:
  source .venv/bin/activate
  export VERTEX_API_KEY="AIza...."
  python homework2_part1_submission.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import gdown
from markitdown import MarkItDown

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

import httpx


FOLDER_ID = "1adYKq7gSSczFP3iikfA8Er-HSZP6VM7D"
DEFAULT_CV_DIR = "downloaded_cvs"
DEFAULT_MCP_URL = "https://ftec5660.ngrok.app/mcp"


def _sim(a: str, b: str) -> float:
    # Use simple ratio to keep dependencies minimal.
    from difflib import SequenceMatcher

    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def _split_location_parts(s: str) -> List[str]:
    """Split a location string like 'Beijing | Hong Kong; Singapore' into parts."""
    if not s:
        return []
    # Keep it simple and robust for noisy LLM extracted strings.
    parts = re.split(r"[|;/\n]+", s)
    out = []
    for p in parts:
        t = " ".join((p or "").strip().split())
        if t:
            out.append(t)
    return out


def _best_sim_any(a_parts: List[str], b: str) -> float:
    if not a_parts or not b:
        return 0.0
    return max(_sim(p, b) for p in a_parts)


def _norm_headline(s: str) -> str:
    t = (s or "").lower().strip()
    t = t.replace("professional", "").strip()
    return " ".join(t.split())


def _is_generic_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return True
    # Very common role words that match too many people.
    generic = {
        "engineer",
        "software engineer",
        "developer",
        "manager",
        "product manager",
        "analyst",
        "consultant",
        "scientist",
        "researcher",
        "intern",
        "professional",
        "specialist",
        "assistant",
    }
    return t in generic or len(t) <= 3


def _token_jaccard(a: str, b: str) -> float:
    stop = {"university", "college", "of", "the", "institute", "school", "department", "faculty"}
    ta = {t for t in re.split(r"[^a-zA-Z0-9]+", (a or "").lower()) if t and t not in stop}
    tb = {t for t in re.split(r"[^a-zA-Z0-9]+", (b or "").lower()) if t and t not in stop}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _extract_first_email(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.I)
    return m.group(0) if m else None


def _norm_year_month(x: Any) -> Optional[dict]:
    """Best-effort normalize many date shapes to {year, month?}."""
    if x is None:
        return None
    if isinstance(x, int):
        return {"year": int(x), "month": None}
    if isinstance(x, dict):
        y = x.get("year") or x.get("y") or x.get("start_year") or x.get("end_year")
        m = x.get("month") or x.get("m") or x.get("start_month") or x.get("end_month")
        try:
            y = int(y) if y is not None else None
        except Exception:
            y = None
        try:
            m = int(m) if m is not None else None
        except Exception:
            m = None
        return {"year": y, "month": m} if y else None
    if isinstance(x, str):
        s = x.strip()
        m = re.search(r"(19\d{2}|20\d{2})(?:\D+(\d{1,2}))?", s)
        if m:
            y = int(m.group(1))
            mm = int(m.group(2)) if m.group(2) else None
            return {"year": y, "month": mm}
    return None


def _extract_linkedin_timeline(profile: Any) -> List[Dict[str, Any]]:
    """Extract (company,title,start,end,is_current) timeline from LinkedIn profile dict."""
    if not isinstance(profile, dict):
        return []
    exps = profile.get("experience")
    if not isinstance(exps, list):
        exps = profile.get("experiences")
    if not isinstance(exps, list):
        return []

    out: List[Dict[str, Any]] = []
    for e in exps:
        if not isinstance(e, dict):
            continue
        company = e.get("company") or e.get("company_name") or e.get("organization")
        title = e.get("title") or e.get("position")

        # Try common variants (some MCP servers return year-only fields)
        start = _norm_year_month(
            e.get("start_date")
            or e.get("startDate")
            or e.get("start")
            or e.get("from")
            or e.get("start_time")
            or e.get("startTime")
            or {"year": e.get("start_year") or e.get("startYear"), "month": e.get("start_month") or e.get("startMonth")}
        )
        end = _norm_year_month(
            e.get("end_date")
            or e.get("endDate")
            or e.get("end")
            or e.get("to")
            or e.get("end_time")
            or e.get("endTime")
            or {"year": e.get("end_year") or e.get("endYear"), "month": e.get("end_month") or e.get("endMonth")}
        )

        dr = e.get("date_range") or e.get("dates")
        if dr is None:
            dr = e.get("dateRange") or e.get("date_range_text")
        if (start is None and end is None) and isinstance(dr, str):
            years = re.findall(r"(19\d{2}|20\d{2})", dr)
            if years:
                start = {"year": int(years[0]), "month": None}
                if re.search(r"present|current|now", dr, flags=re.I):
                    end = None
                elif len(years) >= 2:
                    end = {"year": int(years[1]), "month": None}

        is_current = bool(e.get("is_current") is True) or bool(e.get("isCurrent") is True) or (
            end is None and isinstance(dr, str) and re.search(r"present|current|now", dr, flags=re.I)
        )

        out.append({"company": company, "title": title, "start": start, "end": end, "is_current": is_current})
    return out


def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _safe_json_extract(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _tool_result_to_obj(result: Any) -> Any:
    """
    Convert MCP tool return to python object.
    Common shapes:
    - [{"type":"text","text":"...json..."}]
    - {"structuredContent": {...}} or {"structuredContent":{"result":[...]}}
    - {"content":[...], ...}
    """
    if isinstance(result, dict):
        sc = result.get("structuredContent")
        if isinstance(sc, dict) and "result" in sc:
            return sc["result"]
        if sc is not None:
            return sc

        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                txt = first.get("text") or ""
                try:
                    return json.loads(txt)
                except Exception:
                    return txt
        return result

    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and first.get("type") == "text":
            txt = first.get("text") or ""
            try:
                return json.loads(txt)
            except Exception:
                return txt
    return result


def _download_sample_cvs(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    folder_url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    gdown.download_folder(url=folder_url, output=output_dir, quiet=False, use_cookies=False)


def _load_cvs(cv_dir: str) -> List[Dict[str, str]]:
    md = MarkItDown(enable_plugins=False)
    pdf_files = sorted(
        [f for f in os.listdir(cv_dir) if f.lower().endswith(".pdf")],
        key=lambda x: int("".join(filter(str.isdigit, x))),
    )
    all_cvs = []
    for pdf_name in pdf_files:
        pdf_path = os.path.join(cv_dir, pdf_name)
        result = md.convert(pdf_path)
        all_cvs.append({"file": pdf_name, "text": result.text_content})
    return all_cvs


def _postprocess_score(r: dict) -> float:
    try:
        # Conservative default: if parsing fails / missing, treat as "not verified".
        s = float(r.get("score", 0.45))
    except Exception:
        s = 0.45
    if abs(s - 0.5) < 1e-9:
        # Never return exactly 0.5; keep it on the conservative side.
        s = 0.45 if bool(r.get("discrepancies")) else 0.49

    # If major discrepancy exists, cap score below threshold.
    discs = r.get("discrepancies") or []
    has_major = False
    if isinstance(discs, list):
        for d in discs:
            if isinstance(d, dict) and str(d.get("severity") or "").lower() == "major":
                has_major = True
                break
    if has_major:
        s = min(s, 0.45)

    # If identity isn't confirmed across social channels, be conservative unless confidence is reasonably high.
    # (We rely on the runner's debug signals; this keeps scoring stable even if the LLM drifts.)
    dbg = r.get("_debug") if isinstance(r.get("_debug"), dict) else {}
    if isinstance(dbg, dict):
        ident_conf = dbg.get("social_identity_confidence", dbg.get("linkedin_identity_confidence"))
        ident_ok = dbg.get("social_identity_confirmed", dbg.get("linkedin_identity_confirmed"))
        try:
            ident_conf_f = float(ident_conf) if ident_conf is not None else None
        except Exception:
            ident_conf_f = None

        if ident_ok is False:
            # Low confidence: must not pass.
            if ident_conf_f is None or ident_conf_f < 0.60:
                s = min(s, 0.49)
            else:
                # Medium confidence but not fully confirmed: allow a cautious pass band.
                # Prevent runaway high scores in this ambiguous region.
                s = min(s, 0.60)

    return min(1.0, max(0.0, s))


def evaluate(scores: List[float], groundtruth: List[int], threshold: float = 0.5) -> Dict[str, Any]:
    assert len(scores) == 5
    assert len(groundtruth) == 5
    correct = 0
    decisions = []
    for s, gt in zip(scores, groundtruth):
        pred = 1 if s > threshold else 0
        decisions.append(pred)
        if pred == gt:
            correct += 1
    return {"decisions": decisions, "correct": correct, "total": len(scores), "final_score": correct / len(scores)}


async def _extract_candidate(llm: ChatGoogleGenerativeAI, cv_text: str) -> dict:
    prompt = (
        "请从以下 CV 文本中抽取候选人关键信息，并只输出 JSON：\n"
        "{\n"
        '  "name": "...",\n'
        '  "location_hint": "...",\n'
        '  "current_company": "...",\n'
        '  "current_title": "...",\n'
        '  "current_start_year": 2023,\n'
        '  "highest_degree_level": "phd|masters|bachelors|other|unknown",\n'
        '  "highest_degree_school": "...",\n'
        '  "school": "..." \n'
        "}\n\n"
        "CV 文本：\n" + cv_text
    )
    msg = llm.invoke([HumanMessage(content=prompt)])
    obj = _safe_json_extract(getattr(msg, "content", "") or "")
    if not isinstance(obj, dict):
        first_line = (cv_text.splitlines()[0].strip() if cv_text else "")
        return {
            "name": first_line,
            "location_hint": None,
            "current_company": None,
            "current_title": None,
            "current_start_year": None,
            "highest_degree_level": "unknown",
            "highest_degree_school": None,
            "school": None,
        }
    if isinstance(obj.get("name"), str):
        obj["name"] = obj["name"].split("\n")[0].strip()
    return obj


async def run_agent_for_cv(
    llm: ChatGoogleGenerativeAI,
    tool_by_name: Dict[str, Any],
    cv_text: str,
) -> dict:
    info = await _extract_candidate(llm, cv_text)
    name = (info.get("name") or "").strip()
    location_hint = (info.get("location_hint") or "").strip() or None
    try:
        cv_current_start_year = int(info.get("current_start_year")) if info.get("current_start_year") is not None else None
    except Exception:
        cv_current_start_year = None
    cv_degree_level = str(info.get("highest_degree_level") or "unknown").strip().lower()
    cv_degree_school = (info.get("highest_degree_school") or "") if isinstance(info.get("highest_degree_school"), str) else ""

    improvements: List[str] = []

    async def _ainvoke_with_retry(tool_name: str, args: Dict[str, Any], retries: int = 3) -> Any:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return await tool_by_name[tool_name].ainvoke(args)
            except Exception as e:  # network flakiness (ngrok) / transport read errors
                last_exc = e
                await asyncio.sleep(1.0 * (2**attempt))
        raise last_exc  # type: ignore[misc]

    # LinkedIn forced calls
    linkedin_person_id = None
    linkedin_profile = None
    linkedin_match_score: Optional[float] = None
    try:
        args = {"q": name, "limit": 20, "fuzzy": True}
        if location_hint:
            args["location"] = location_hint
        li_search = _tool_result_to_obj(await _ainvoke_with_retry("search_linkedin_people", args))
        if isinstance(li_search, dict) and "result" in li_search:
            li_search = li_search["result"]
        if (not isinstance(li_search, list)) or (len(li_search) == 0):
            li_search = _tool_result_to_obj(
                await _ainvoke_with_retry("search_linkedin_people", {"q": name, "limit": 20, "fuzzy": True})
            )
            if isinstance(li_search, dict) and "result" in li_search:
                li_search = li_search["result"]
        if isinstance(li_search, list) and li_search:
            # 先按搜索摘要字段做一次粗排（headline/location/exact）
            ranked: List[Dict[str, Any]] = []
            # NOTE: extraction doesn't reliably provide "headline"; fall back to current_title for weak signal.
            cv_headline = _norm_headline(str(info.get("headline") or info.get("current_title") or ""))
            cv_loc = str(location_hint or "")

            tmp: List[tuple] = []
            for c in li_search:
                if not isinstance(c, dict) or not isinstance(c.get("id"), int):
                    continue
                s = 1.0  # 同名列表里 name 基本一致，给常数
                h = _norm_headline(str(c.get("headline") or ""))
                if cv_headline and h and cv_headline == h:
                    s += 0.8
                else:
                    s += 0.2 * _sim(cv_headline, h)
                if cv_loc:
                    s += 0.25 * _sim(cv_loc, str(c.get("location") or ""))
                if str(c.get("match_type") or "").lower() == "exact":
                    s += 0.05
                tmp.append((s, c))
            tmp.sort(key=lambda x: x[0], reverse=True)
            ranked = [c for _, c in tmp]

            # 对 topK 拉 profile，再用 current company/title + education 做二次消歧
            # (top3 is sometimes too small; it can miss the correct person when location is common)
            best_pid = None
            best_prof = None
            best_score = -1.0
            K = 6
            cv_title_txt = str(info.get("current_title") or "")
            cv_title_generic = _is_generic_title(cv_title_txt)
            for cand in ranked[:K]:
                pid = cand.get("id")
                prof = _tool_result_to_obj(await _ainvoke_with_retry("get_linkedin_profile", {"person_id": pid}))
                if isinstance(prof, dict) and "result" in prof:
                    prof = prof["result"]
                if not isinstance(prof, dict):
                    continue

                s2 = 0.0
                # headline（弱惩罚，避免同名误选）
                li_h = _norm_headline(str(prof.get("headline") or ""))
                if cv_headline and li_h and cv_headline == li_h:
                    s2 += 1.0
                elif cv_headline and li_h:
                    s2 -= 0.2

                # current company/title（只加分，不强惩罚）
                cur = None
                if isinstance(prof.get("experience"), list):
                    cur = next(
                        (e for e in prof["experience"] if isinstance(e, dict) and e.get("is_current") is True),
                        None,
                    )
                if cur and info.get("current_company"):
                    s2 += 0.6 * _sim(str(info.get("current_company")), str(cur.get("company") or ""))
                if cur and info.get("current_title"):
                    title_sim = _sim(str(info.get("current_title")), str(cur.get("title") or ""))
                    # Generic titles (e.g. "Engineer") are weak evidence; down-weight heavily.
                    s2 += (0.08 if cv_title_generic else 0.3) * title_sim

                # school token overlap
                if info.get("school") and isinstance(prof.get("education"), list):
                    best_school = 0.0
                    for ed in prof["education"]:
                        if isinstance(ed, dict):
                            best_school = max(best_school, _token_jaccard(str(info.get("school")), str(ed.get("school") or "")))
                    s2 += 0.4 * best_school
                if cv_degree_school and isinstance(prof.get("education"), list):
                    best_school2 = 0.0
                    for ed in prof["education"]:
                        if isinstance(ed, dict):
                            best_school2 = max(best_school2, _token_jaccard(cv_degree_school, str(ed.get("school") or "")))
                    s2 += 0.5 * best_school2

                if s2 > best_score:
                    best_score = s2
                    best_pid = pid
                    best_prof = prof

            # 置信度门槛：太低就认为同名不确定（避免误匹配导致一堆 major）
            # If CV title is generic, require a bit more evidence to avoid false matches.
            min_pick = 0.42 if cv_title_generic else 0.3
            if best_pid is not None and best_score >= min_pick:
                linkedin_person_id = best_pid
                linkedin_profile = best_prof
                linkedin_match_score = float(best_score)
            else:
                improvements.append("LinkedIn 同名过多/置信度不足，未选定 profile（避免误匹配）。")
        else:
            improvements.append("LinkedIn 未找到匹配 profile（已尝试带/不带 location 搜索）。")
    except Exception as e:
        improvements.append(f"LinkedIn tool 调用失败: {e}")

    # Facebook forced calls
    facebook_user_id = None
    facebook_profile = None
    facebook_match_score: Optional[float] = None
    try:
        fb_search = _tool_result_to_obj(
            await _ainvoke_with_retry("search_facebook_users", {"q": name, "limit": 20, "fuzzy": True})
        )
        if isinstance(fb_search, dict) and "result" in fb_search:
            fb_search = fb_search["result"]
        if isinstance(fb_search, list) and fb_search:
            # 先按 display_name + location 做粗排，再对 top3 拉 profile 用 company/location 复核
            scored: List[tuple] = []
            for c in fb_search:
                if not isinstance(c, dict) or not isinstance(c.get("id"), int):
                    continue
                s = _sim(name, str(c.get("display_name") or ""))
                if location_hint:
                    s += 0.2 * _sim(location_hint, f"{c.get('city','')}, {c.get('country','')}")
                scored.append((s, c))
            scored.sort(key=lambda x: x[0], reverse=True)

            best_uid = None
            best_prof = None
            best_score = -1.0
            for base, cand in scored[:3]:
                uid = cand.get("id")
                prof = _tool_result_to_obj(await _ainvoke_with_retry("get_facebook_profile", {"user_id": uid}))
                if isinstance(prof, dict) and "result" in prof:
                    prof = prof["result"]
                if not isinstance(prof, dict):
                    continue
                s2 = float(base)
                if info.get("current_company") and prof.get("current_company"):
                    s2 += 0.4 * _sim(str(info.get("current_company")), str(prof.get("current_company")))
                if location_hint:
                    s2 += 0.2 * _sim(location_hint, f"{prof.get('city','')}, {prof.get('country','')}")
                if s2 > best_score:
                    best_score = s2
                    best_uid = uid
                    best_prof = prof

            if best_uid is not None and best_score >= 0.5:
                facebook_user_id = best_uid
                facebook_profile = best_prof
                facebook_match_score = float(best_score)
            else:
                improvements.append("Facebook 同名过多/置信度不足，未选定 profile（避免误匹配）。")
        else:
            improvements.append("Facebook 未找到匹配 profile（search_facebook_users 返回空）。")
    except Exception as e:
        improvements.append(f"Facebook tool 调用失败: {e}")

    linkedin_experience_timeline = _extract_linkedin_timeline(linkedin_profile)

    # CV location can be multi-valued like "Beijing | Hong Kong"; compare against LinkedIn location robustly.
    cv_loc_parts = _split_location_parts(str(location_hint or ""))

    def _timeline_latest_end_year(tl: List[Dict[str, Any]]) -> Optional[int]:
        ys: List[int] = []
        for e in tl:
            end = e.get("end")
            if isinstance(end, dict) and isinstance(end.get("year"), int):
                ys.append(end["year"])
        return max(ys) if ys else None

    def _timeline_has_current(tl: List[Dict[str, Any]]) -> bool:
        return any(bool(e.get("is_current")) for e in tl)

    li_latest_end = _timeline_latest_end_year(linkedin_experience_timeline)
    li_has_current = _timeline_has_current(linkedin_experience_timeline)
    linkedin_possible_outdated = False
    linkedin_outdated_reason = ""
    if (cv_current_start_year is not None) and (not li_has_current) and (li_latest_end is not None):
        # 如果 CV 说当前工作从某年开始，但 LinkedIn 最晚结束年份明显早于该年份，优先认为社媒陈旧/未更新
        if li_latest_end <= cv_current_start_year - 1:
            linkedin_possible_outdated = True
            linkedin_outdated_reason = f"CV current_start_year={cv_current_start_year}，LinkedIn latest_end_year={li_latest_end} 且无 current 记录"

    # ---- Identity disambiguation (confirm LinkedIn profile is the same person)
    cv_email = _extract_first_email(cv_text or "")
    li_location = ""
    li_headline = ""
    li_email = ""
    li_edu_schools: List[str] = []
    if isinstance(linkedin_profile, dict):
        li_location = str(linkedin_profile.get("location") or linkedin_profile.get("city") or "")
        li_headline = str(linkedin_profile.get("headline") or "")
        li_email = str(
            linkedin_profile.get("email")
            or linkedin_profile.get("contact_email")
            or linkedin_profile.get("contactEmail")
            or ""
        )
        if isinstance(linkedin_profile.get("education"), list):
            for ed in linkedin_profile["education"]:
                if isinstance(ed, dict) and ed.get("school"):
                    li_edu_schools.append(str(ed.get("school") or ""))

    # Strong anchors: email / education / location
    email_match = bool(cv_email and li_email and cv_email.lower() == li_email.lower())
    edu_match = False
    if cv_degree_school:
        edu_match = any(_token_jaccard(cv_degree_school, s) >= 0.5 for s in li_edu_schools)
    elif isinstance(info.get("school"), str) and info.get("school"):
        edu_match = any(_token_jaccard(str(info.get("school")), s) >= 0.5 for s in li_edu_schools)

    # Location match: allow any CV location part to match LI location.
    best_loc_sim = _best_sim_any(cv_loc_parts, li_location)
    location_match = bool(cv_loc_parts and li_location and best_loc_sim >= 0.6)

    # Weak anchors: headline similarity, match_score
    headline_match = bool(info.get("current_title") and li_headline and _sim(str(info.get("current_title")), li_headline) >= 0.55)
    match_score_term = min(1.0, max(0.0, float(linkedin_match_score or 0.0)))  # already roughly [0,~2], clamp

    # Medium anchors: current job match (company/title) against LinkedIn current experience(s)
    cv_cur_company = str(info.get("current_company") or "").strip()
    cv_cur_title = str(info.get("current_title") or "").strip()
    cur_company_sim = 0.0
    cur_title_sim = 0.0
    if cv_cur_company or cv_cur_title:
        for e in linkedin_experience_timeline:
            if not isinstance(e, dict) or not bool(e.get("is_current")):
                continue
            if cv_cur_company and e.get("company"):
                cur_company_sim = max(cur_company_sim, _sim(cv_cur_company, str(e.get("company") or "")))
            if cv_cur_title and e.get("title"):
                cur_title_sim = max(cur_title_sim, _sim(cv_cur_title, str(e.get("title") or "")))
    # Title-only matches can be too generic ("Engineer" etc). Be stricter when company is missing.
    current_job_match = bool(
        (cv_cur_company and cur_company_sim >= 0.7)
        or ((not cv_cur_company) and cv_cur_title and cur_title_sim >= 0.9)
        or (cv_cur_company and cv_cur_title and cur_company_sim >= 0.6 and cur_title_sim >= 0.75)
    )

    strong_hits = int(email_match) + int(edu_match) + int(location_match)

    linkedin_identity_confidence = 0.0
    # email is rare but very strong
    if email_match:
        linkedin_identity_confidence += 0.6
    if edu_match:
        linkedin_identity_confidence += 0.4
    if location_match:
        linkedin_identity_confidence += 0.3
    # current job is a medium-strength anchor (can be noisy, but helps when email missing)
    if current_job_match:
        linkedin_identity_confidence += 0.25
    if headline_match:
        linkedin_identity_confidence += 0.15
    linkedin_identity_confidence += 0.15 * match_score_term
    linkedin_identity_confidence = min(1.0, linkedin_identity_confidence)

    # Confirm identity if:
    # - at least 2 strong anchors (email/edu/location), OR
    # - edu+current_job (common when email hidden, location noisy), OR
    # - location+current_job with some match_score support, OR
    # - very high overall confidence
    linkedin_identity_confirmed = bool(
        strong_hits >= 2
        or (edu_match and current_job_match)
        or (location_match and current_job_match and match_score_term >= 0.4)
        or (linkedin_identity_confidence >= 0.82)
    )
    identity_notes = []
    if cv_email:
        identity_notes.append(f"cv_email={cv_email}, li_email={'present' if li_email else 'missing'}, email_match={email_match}")
    if cv_degree_school or info.get("school"):
        identity_notes.append(f"edu_match={edu_match} (cv_school={cv_degree_school or info.get('school')}, li_edu_schools={li_edu_schools[:3]})")
    if location_hint:
        identity_notes.append(
            f"location_match={location_match} (best_sim={best_loc_sim:.2f}, cv_location={location_hint}, li_location={li_location})"
        )
    if cv_cur_company or cv_cur_title:
        identity_notes.append(
            f"current_job_match={current_job_match} (company_sim={cur_company_sim:.2f}, title_sim={cur_title_sim:.2f}, cv_company={cv_cur_company}, cv_title={cv_cur_title})"
        )
    identity_notes.append(f"headline_match={headline_match} (cv_title={info.get('current_title')}, li_headline={li_headline})")
    identity_notes.append(f"match_score={linkedin_match_score}")

    # ---- Facebook identity disambiguation (acts as an alternative evidence channel)
    fb_location = ""
    fb_company = ""
    fb_title = ""
    fb_school = ""
    fb_degree = ""
    if isinstance(facebook_profile, dict):
        fb_location = _norm_str(facebook_profile.get("location") or f"{facebook_profile.get('city','')}, {facebook_profile.get('country','')}")
        fb_company = _norm_str(
            facebook_profile.get("current_company")
            or facebook_profile.get("company")
            or facebook_profile.get("employer")
            or facebook_profile.get("workplace")
        )
        fb_title = _norm_str(
            facebook_profile.get("current_title")
            or facebook_profile.get("title")
            or facebook_profile.get("position")
            or facebook_profile.get("job_title")
        )
        fb_school = _norm_str(
            facebook_profile.get("school")
            or facebook_profile.get("education_school")
            or facebook_profile.get("university")
            or facebook_profile.get("college")
        )
        fb_degree = _norm_str(facebook_profile.get("degree") or facebook_profile.get("education_level") or facebook_profile.get("education"))

    fb_best_loc_sim = _best_sim_any(cv_loc_parts, fb_location)
    fb_location_match = bool(cv_loc_parts and fb_location and fb_best_loc_sim >= 0.6)

    fb_company_sim = _sim(_norm_str(info.get("current_company")), fb_company) if (info.get("current_company") and fb_company) else 0.0
    fb_title_sim = _sim(_norm_str(info.get("current_title")), fb_title) if (info.get("current_title") and fb_title) else 0.0
    fb_current_job_match = bool(
        (info.get("current_company") and fb_company and fb_company_sim >= 0.7)
        or ((not info.get("current_company")) and info.get("current_title") and fb_title and fb_title_sim >= 0.9)
        or (info.get("current_company") and info.get("current_title") and fb_company and fb_title and fb_company_sim >= 0.6 and fb_title_sim >= 0.75)
    )

    fb_edu_match = False
    if cv_degree_school and fb_school:
        fb_edu_match = _token_jaccard(cv_degree_school, fb_school) >= 0.5
    elif isinstance(info.get("school"), str) and info.get("school") and fb_school:
        fb_edu_match = _token_jaccard(str(info.get("school")), fb_school) >= 0.5

    # facebook_match_score can exceed 1.0; normalize roughly to [0,1]
    try:
        fb_ms = float(facebook_match_score or 0.0)
    except Exception:
        fb_ms = 0.0
    fb_match_score_term = min(1.0, max(0.0, fb_ms / 1.6))

    facebook_identity_confidence = 0.0
    if fb_edu_match:
        facebook_identity_confidence += 0.4
    if fb_location_match:
        facebook_identity_confidence += 0.3
    if fb_current_job_match:
        facebook_identity_confidence += 0.35
    facebook_identity_confidence += 0.15 * fb_match_score_term
    facebook_identity_confidence = min(1.0, facebook_identity_confidence)

    facebook_identity_confirmed = bool(
        (fb_edu_match and fb_location_match)
        or (fb_edu_match and fb_current_job_match)
        or (facebook_identity_confidence >= 0.78)
    )

    # Overall identity: prefer whichever channel is more confident.
    social_identity_confirmed = bool(linkedin_identity_confirmed or facebook_identity_confirmed)
    social_identity_confidence = float(max(linkedin_identity_confidence, facebook_identity_confidence))

    identity_notes.append(
        f"facebook_identity_confirmed={facebook_identity_confirmed} (conf={facebook_identity_confidence:.2f}, loc_match={fb_location_match}, edu_match={fb_edu_match}, cur_job_match={fb_current_job_match})"
    )

    final_prompt = (
        "你是 CV 核验智能体。你已经拿到 CV 文本、以及通过 MCP tools 获取的社媒证据。\n"
        "请对照 CV 与社媒证据，输出最终 JSON（只输出 JSON，不要 markdown）。\n\n"
        "关键核对原则（必须遵守）：\n"
        "0) 身份确认优先于差异判断：只有当你能合理确认某个社媒档案与 CV 是同一人时，才把该平台差异作为强证据。\n"
        "   - 强锚点（至少命中 2 个更可信）：邮箱/联系方式一致、教育学校一致、地理位置一致。\n"
        "   - 弱锚点只能辅助：同名、headline 模糊相似、单个公司/职位。\n"
        "   - 如果某个平台无法确认同一人（*_identity_confirmed=False 或 *_identity_confidence<0.75），该平台的 company/title/education/experience 差异一律写 improvement（提示“可能同名误匹配/未更新，需人工复核”），不要判 major。\n"
        "1) 核对公司/职位时，必须结合“对应时间段”一起考虑。只有当 LinkedIn 的经历时间段覆盖/重叠 CV 的对应年份时，才把 company/title 不一致作为 discrepancy。\n"
        "2) 如果 LinkedIn 时间线明显陈旧/缺失（例如没有 current，且 latest_end_year 早于 CV current_start_year），则 company/title 不一致应当写成 improvement（提示“可能未更新”），不要判 major。\n"
        "3) 教育：若 CV 明确写最高学历为 PhD，但 LinkedIn/Facebook 仅显示 MSc/BSc/本科（且 profile 匹配置信度足够），应当判为 major discrepancy（学历层级差异属于高风险）。\n\n"
        "评分规则（必须严格按以下区间打分）：\n"
        "- 不要输出刚好 0.5。\n"
        "- 若 social_identity_confirmed=False 且 social_identity_confidence<0.60：代表证据不足/无法核验同一人，score 必须 <=0.49（偏保守，宁可人工复核）。\n"
        "- 若 social_identity_confirmed=False 且 0.60<=social_identity_confidence<0.75：代表部分证据支持但仍有不确定，score 给 0.52~0.60，并在 improvements 写明“建议人工抽查身份”。\n"
        "- 若 identity_confirmed=True 且无 discrepancies：score 给 0.75~0.90。\n"
        "- 若 identity_confirmed=True 且只有 minor discrepancies：score 给 0.55~0.70。\n"
        "- 若存在任何 major discrepancy：score 必须 <=0.45。\n\n"
        "输出 JSON 格式：\n"
        "{\n"
        '  "score": 0.0,\n'
        '  "linkedin_person_id": null,\n'
        '  "facebook_user_id": null,\n'
        '  "discrepancies": [{"severity":"major|minor","field":"...","cv":"...","social":"...","evidence":"..."}],\n'
        '  "improvements": ["..."]\n'
        "}\n\n"
        "CV 文本：\n" + cv_text + "\n\n"
        "提取信息：\n" + json.dumps(info, ensure_ascii=False) + "\n\n"
        f"LinkedIn person_id: {linkedin_person_id}\n"
        f"LinkedIn match_score: {linkedin_match_score}\n"
        f"LinkedIn identity_confirmed: {linkedin_identity_confirmed}\n"
        f"LinkedIn identity_confidence: {linkedin_identity_confidence}\n"
        f"Facebook identity_confirmed: {facebook_identity_confirmed}\n"
        f"Facebook identity_confidence: {facebook_identity_confidence}\n"
        f"Social identity_confirmed: {social_identity_confirmed}\n"
        f"Social identity_confidence: {social_identity_confidence}\n"
        "LinkedIn identity_notes:\n" + "\n".join(identity_notes) + "\n\n"
        "LinkedIn experience_timeline（normalized）：\n"
        + json.dumps(linkedin_experience_timeline, ensure_ascii=False)
        + "\n\n"
        f"LinkedIn possible_outdated: {linkedin_possible_outdated} ({linkedin_outdated_reason})\n\n"
        "LinkedIn profile:\n" + json.dumps(linkedin_profile, ensure_ascii=False) + "\n\n"
        f"Facebook user_id: {facebook_user_id}\n"
        f"Facebook match_score: {facebook_match_score}\n"
        "Facebook profile:\n" + json.dumps(facebook_profile, ensure_ascii=False) + "\n\n"
        "已知限制/提示：\n" + "\n".join(improvements) + "\n"
    )
    msg = llm.invoke([HumanMessage(content=final_prompt)])
    out = _safe_json_extract(getattr(msg, "content", "") or "") or {}

    out["linkedin_person_id"] = out.get("linkedin_person_id") or linkedin_person_id
    out["facebook_user_id"] = out.get("facebook_user_id") or facebook_user_id
    out_imps = out.get("improvements") if isinstance(out.get("improvements"), list) else []
    out["improvements"] = list(dict.fromkeys(out_imps + improvements))

    # Attach debug evidence for file output (doesn't change scoring semantics)
    out["_debug"] = {
        "cv_current_start_year": cv_current_start_year,
        "cv_highest_degree_level": cv_degree_level,
        "cv_highest_degree_school": cv_degree_school,
        "cv_email": cv_email,
        "linkedin_match_score": linkedin_match_score,
        "linkedin_identity_confirmed": linkedin_identity_confirmed,
        "linkedin_identity_confidence": linkedin_identity_confidence,
        "linkedin_identity_notes": identity_notes,
        "facebook_identity_confirmed": facebook_identity_confirmed,
        "facebook_identity_confidence": facebook_identity_confidence,
        "social_identity_confirmed": social_identity_confirmed,
        "social_identity_confidence": social_identity_confidence,
        "identity_signals": {
            "email_match": email_match,
            "edu_match": edu_match,
            "location_match": location_match,
            "best_loc_sim": best_loc_sim,
            "current_job_match": current_job_match,
            "cur_company_sim": cur_company_sim,
            "cur_title_sim": cur_title_sim,
            "match_score_term": match_score_term,
            "strong_hits": strong_hits,
            "fb_location_match": fb_location_match,
            "fb_best_loc_sim": fb_best_loc_sim,
            "fb_edu_match": fb_edu_match,
            "fb_current_job_match": fb_current_job_match,
            "fb_company_sim": fb_company_sim,
            "fb_title_sim": fb_title_sim,
            "fb_match_score_term": fb_match_score_term,
        },
        "facebook_match_score": facebook_match_score,
        "linkedin_experience_timeline": linkedin_experience_timeline,
        "linkedin_experience_raw_first3": (
            linkedin_profile.get("experience")[:3]
            if isinstance(linkedin_profile, dict) and isinstance(linkedin_profile.get("experience"), list)
            else None
        ),
        "linkedin_possible_outdated": linkedin_possible_outdated,
        "linkedin_outdated_reason": linkedin_outdated_reason,
    }
    return out


async def async_main() -> None:
    # Avoid proxy-related MCP connection failures (httpx honors *_PROXY env vars).
    for k in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ]:
        os.environ.pop(k, None)

    # Also disable macOS/system proxy discovery inside httpx by forcing trust_env=False
    # via langchain-mcp-adapters `httpx_client_factory`.
    def _httpx_client_factory(
        headers: Dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        # follow_redirects=True is required because some ngrok setups redirect http->https (307),
        # and MCP's streamable-http transport uses streaming requests that otherwise won't follow.
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            auth=auth,
            trust_env=False,
            follow_redirects=True,
        )

    api_key = os.environ.get("VERTEX_API_KEY")
    if not api_key:
        raise SystemExit("Missing VERTEX_API_KEY. Please export VERTEX_API_KEY before running.")
    api_key = api_key.strip()
    if any(ord(ch) > 127 for ch in api_key):
        raise SystemExit("VERTEX_API_KEY contains non-ascii characters. Re-export it cleanly.")

    # ensure CVs exist (match the notebook behavior: always use gdown -> downloaded_cvs)
    cv_dir = DEFAULT_CV_DIR
    if not os.path.isdir(cv_dir) or not any(x.lower().endswith(".pdf") for x in os.listdir(cv_dir)):
        _download_sample_cvs(cv_dir)

    all_cvs = _load_cvs(cv_dir)

    # MCP client + tools
    mcp_url = os.environ.get("MCP_URL", DEFAULT_MCP_URL).strip()

    async def _load_tools_with_url(url: str) -> List[Any]:
        client = MultiServerMCPClient(
            {
                "social_graph": {
                    "transport": "http",
                    "url": url,
                    "headers": {"ngrok-skip-browser-warning": "true"},
                    "httpx_client_factory": _httpx_client_factory,
                    # Some servers/ngrok setups don't support DELETE termination; avoid noisy warning.
                    "terminate_on_close": False,
                }
            }
        )
        # ngrok/network can be flaky; retry tool loading a few times
        tools: List[Any] = []
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                tools = await client.get_tools()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                await asyncio.sleep(1.0 * (2**attempt))
        if last_exc is not None:
            raise last_exc
        return tools

    def _swap_scheme(url: str) -> str:
        if url.startswith("https://"):
            return "http://" + url[len("https://") :]
        if url.startswith("http://"):
            return "https://" + url[len("http://") :]
        return url

    # ngrok/MCP can be flaky; also some environments block one scheme but not the other.
    # Always probe both http/https variants (when applicable), even if MCP_URL is explicitly set.
    candidate_urls: List[str] = [mcp_url]
    alt = _swap_scheme(mcp_url)
    if alt != mcp_url:
        candidate_urls.append(alt)

    tools: List[Any] = []
    last_exc: Exception | None = None
    for url in candidate_urls:
        try:
            tools = await _load_tools_with_url(url)
            mcp_url = url
            last_exc = None
            break
        except Exception as e:
            last_exc = e
    if last_exc is not None:
        raise last_exc
    tool_by_name = {t.name: t for t in tools}

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,
    )

    results: List[dict] = []
    for item in all_cvs:
        r = await run_agent_for_cv(llm, tool_by_name, item["text"])
        r["file"] = item["file"]
        r["_debug"] = r.get("_debug") or {}
        if isinstance(r["_debug"], dict):
            r["_debug"]["mcp_url"] = mcp_url
        results.append(r)

    scores = [_postprocess_score(r) for r in results]
    print("scores =", scores)

    # print summary + write file
    out_txt = "llm_results.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for r in results:
            final_s = _postprocess_score(r)
            f.write("=" * 80 + "\n")
            f.write(str(r.get("file")) + "\n")
            f.write("score_raw: " + str(r.get("score")) + "\n")
            f.write("score_final: " + str(final_s) + "\n")
            f.write(f"LinkedIn: {r.get('linkedin_person_id')} Facebook: {r.get('facebook_user_id')}\n\n")
            dbg = r.get("_debug") if isinstance(r.get("_debug"), dict) else {}
            if dbg:
                f.write("[debug]\n")
                f.write("cv_current_start_year: " + str(dbg.get("cv_current_start_year")) + "\n")
                f.write("cv_highest_degree_level: " + str(dbg.get("cv_highest_degree_level")) + "\n")
                f.write("cv_highest_degree_school: " + str(dbg.get("cv_highest_degree_school")) + "\n")
                f.write("cv_email: " + str(dbg.get("cv_email")) + "\n")
                f.write("linkedin_match_score: " + str(dbg.get("linkedin_match_score")) + "\n")
                f.write("linkedin_identity_confirmed: " + str(dbg.get("linkedin_identity_confirmed")) + "\n")
                f.write("linkedin_identity_confidence: " + str(dbg.get("linkedin_identity_confidence")) + "\n")
                if isinstance(dbg.get("identity_signals"), dict):
                    sig = dbg.get("identity_signals") or {}
                    f.write("identity_signals: " + json.dumps(sig, ensure_ascii=False) + "\n")
                if isinstance(dbg.get("linkedin_identity_notes"), list) and dbg.get("linkedin_identity_notes"):
                    f.write("linkedin_identity_notes:\n")
                    for note in dbg.get("linkedin_identity_notes") or []:
                        f.write("- " + str(note) + "\n")
                f.write("facebook_match_score: " + str(dbg.get("facebook_match_score")) + "\n")
                if dbg.get("mcp_url") is not None:
                    f.write("mcp_url: " + str(dbg.get("mcp_url")) + "\n")
                f.write(
                    "linkedin_possible_outdated: "
                    + str(dbg.get("linkedin_possible_outdated"))
                    + " ("
                    + str(dbg.get("linkedin_outdated_reason"))
                    + ")\n"
                )
                f.write("linkedin_experience_timeline:\n")
                f.write(json.dumps(dbg.get("linkedin_experience_timeline") or [], ensure_ascii=False) + "\n\n")
                if dbg.get("linkedin_experience_raw_first3") is not None:
                    f.write("linkedin_experience_raw_first3:\n")
                    f.write(json.dumps(dbg.get("linkedin_experience_raw_first3") or [], ensure_ascii=False) + "\n\n")
            f.write("[discrepancies]\n")
            for d in r.get("discrepancies", []) or []:
                f.write("- " + json.dumps(d, ensure_ascii=False) + "\n")
            f.write("\n[improvements]\n")
            for s in r.get("improvements", []) or []:
                f.write("- " + str(s) + "\n")
            f.write("\n")
    print("Saved full output to", out_txt)

    # optional: evaluation on starter groundtruth (for sanity/debug)
    if str(os.environ.get("RUN_SANITY_EVAL") or "").strip().lower() in {"1", "true", "yes"}:
        groundtruth = [1, 1, 1, 0, 0]
        print("sanity evaluate:", evaluate(scores, groundtruth))


if __name__ == "__main__":
    asyncio.run(async_main())

