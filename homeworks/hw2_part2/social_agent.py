"""
Moltbook Autonomous Agent (FTEC5660 HW2 Part2)

Only agent-related code:
- Reads https://www.moltbook.com/skill.md
- Fetches m/ftec5660 feed
- Uses an LLM to decide and execute actions: subscribe / upvote / comment / stop
- Enforces anti-spam constraints with a local state file
- Forces English-only comments

Auth:
- Moltbook: env MOLTBOOK_API_KEY or ~/.config/moltbook/credentials.json {"api_key": "..."}
- LLM: env VERTEX_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY (API key for Gemini Developer API)

Note: Do NOT paste API keys into chats or screenshots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# skill.md 强制要求：必须带 www，否则重定向可能丢 Authorization
BASE_URL = "https://www.moltbook.com/api/v1"
SKILL_MD_URL = "https://www.moltbook.com/skill.md"

# 作业社区与目标帖
TARGET_SUBMOLT = "ftec5660"
TARGET_POST_ID = "47ff50f3-8255-4dee-87f4-2c3637c7351c"
REQUIRED_COMMENT_POST_ID = TARGET_POST_ID

# state：避免重复互动/刷屏（默认放在脚本同目录）
DEFAULT_STATE_PATH = Path(__file__).with_name("moltbook_agent_state.json")

# 必做评论（英文，简短且与欢迎帖相关）
DEFAULT_REQUIRED_COMMENT_EN = "Thanks for setting up this space for FTEC5660—looking forward to learning and sharing here!"


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _truncate(s: str, n: int = 240) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 3] + "..."


def _contains_cjk(text: str) -> bool:
    """粗略检测中日韩字符，用于强制英文评论要求。"""
    if not text:
        return False
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))

def _is_english_comment(text: str) -> bool:
    """本作业里把“英文评论”近似为：不包含 CJK 字符。"""
    return bool(text and isinstance(text, str) and (not _contains_cjk(text)))


def _now_ts() -> float:
    return time.time()


def _iso_utc(ts: Optional[float] = None) -> str:
    import datetime as _dt

    if ts is None:
        ts = _now_ts()
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_moltbook_api_key(override: Optional[str] = None) -> str:
    if override and override.strip():
        return override.strip()

    api_key = os.getenv("MOLTBOOK_API_KEY")
    if api_key:
        return api_key.strip()

    cred_path = Path("~/.config/moltbook/credentials.json").expanduser()
    if cred_path.exists():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            api_key = (data.get("api_key") or "").strip()
            if api_key:
                return api_key
        except Exception:
            pass

    raise RuntimeError(
        "未找到 Moltbook API key。请设置环境变量 MOLTBOOK_API_KEY，"
        "或在 ~/.config/moltbook/credentials.json 写入 {\"api_key\": \"...\"}。"
    )


def _load_llm_api_key(override: Optional[str] = None) -> Optional[str]:
    """读取 Gemini Developer API key（支持 API key 的那条路径）。"""
    if override and override.strip():
        return override.strip()
    return (
        os.getenv("VERTEX_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip() or None


def _safe_headers(api_key: str) -> dict[str, str]:
    if not BASE_URL.startswith("https://www.moltbook.com/api/v1"):
        raise RuntimeError("BASE_URL 不安全：必须是 https://www.moltbook.com/api/v1")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _http_get_text(url: str, timeout: int = 20) -> str:
    req = Request(url=url, method="GET", headers={"Accept": "text/plain"})
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read() or b""
        return body.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class MoltbookClient:
    api_key: str

    @property
    def headers(self) -> dict[str, str]:
        return _safe_headers(self.api_key)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        timeout: int = 20,
    ) -> dict:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{BASE_URL}{path}"
        if params:
            qs = urlencode({k: v for k, v in params.items() if v is not None})
            if qs:
                url = f"{url}?{qs}"

        data: Optional[bytes] = None
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")

        req = Request(url=url, method=method.upper(), headers=self.headers, data=data)

        def _parse_json_bytes(b: bytes, status: Optional[int] = None) -> dict:
            try:
                return json.loads(b.decode("utf-8"))
            except Exception:
                return {
                    "success": False,
                    "error": f"Non-JSON response{'' if status is None else f' (status={status})'}",
                    "raw": b.decode("utf-8", errors="replace"),
                }

        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read() or b""
                return _parse_json_bytes(body, getattr(resp, "status", None))
        except HTTPError as e:
            body = e.read() or b""
            parsed = _parse_json_bytes(body, getattr(e, "code", None))
            if isinstance(parsed, dict):
                parsed.setdefault("http_status", getattr(e, "code", None))
            return parsed
        except URLError as e:
            return {"success": False, "error": f"Network error: {e}"}

    def status(self) -> dict:
        return self.request("GET", "/agents/status")

    def me(self) -> dict:
        return self.request("GET", "/agents/me")

    def subscribe(self, submolt: str) -> dict:
        return self.request("POST", f"/submolts/{submolt}/subscribe")

    def upvote_post(self, post_id: str) -> dict:
        return self.request("POST", f"/posts/{post_id}/upvote")

    def comment_post(self, post_id: str, content: str) -> dict:
        return self.request("POST", f"/posts/{post_id}/comments", json_body={"content": content})

    def verify(self, verification_code: str, answer: str) -> dict:
        return self.request("POST", "/verify", json_body={"verification_code": verification_code, "answer": answer})

    def submolt_feed(self, submolt: str, sort: str = "new", limit: int = 10) -> dict:
        return self.request("GET", f"/submolts/{submolt}/feed", params={"sort": sort, "limit": limit})

    def get_post(self, post_id: str) -> dict:
        return self.request("GET", f"/posts/{post_id}")


def _rewrite_to_english(*, llm: Any, llm2: Any, text: str) -> str:
    if not text:
        return ""
    if not _contains_cjk(text):
        return text
    prompt = (
        "Rewrite the following into concise, friendly English only. "
        "Return ONLY the rewritten text (no quotes, no extra words):\n\n"
        f"{text}\n"
    )
    try:
        resp = llm.invoke(prompt)
        out = (resp.content or "").strip()
    except Exception:
        resp = llm2.invoke(prompt)
        out = (resp.content or "").strip()
    return "" if _contains_cjk(out) else out


def _solve_verification_with_llm(*, llm: Any, llm2: Any, challenge: str) -> Optional[str]:
    """
    让 LLM 解验证码题并输出两位小数（纯数字字符串）。
    适配 Moltbook 常见的“混淆大小写/插入符号”的题面。
    """
    if not challenge:
        return None
    prompt = (
        "Solve the math problem and reply with ONLY the number with 2 decimal places.\n\n"
        f"Problem:\n{challenge}\n"
    )
    try:
        resp = llm.invoke(prompt)
        text = (resp.content or "").strip()
    except Exception:
        resp = llm2.invoke(prompt)
        text = (resp.content or "").strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    return f"{float(m.group(1)):.2f}"


def autonomous_agent_run(
    *,
    client: MoltbookClient,
    llm_api_key: str,
    state_path: Path,
    max_actions: int,
    dry_run: bool,
    verbose: bool,
) -> dict[str, Any]:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        return {
            "success": False,
            "error": "Missing dependency: langchain_google_genai",
            "hint": "请在 venv 里安装：pip install langchain-google-genai langchain-core",
            "detail": str(e),
        }

    state = _load_state(state_path)
    state.setdefault("subscribed_submolts", [])
    state.setdefault("upvoted_post_ids", [])
    state.setdefault("commented_post_ids", [])
    state.setdefault("pending_verifications", {})
    state.setdefault("last_action_at", None)

    # 读取 skill.md + feed（作为“已读文档 + 已观察环境”的证据）
    skill_md = _http_get_text(SKILL_MD_URL, timeout=20)
    feed = client.submolt_feed(TARGET_SUBMOLT, sort="new", limit=10)
    required_post_snapshot = client.get_post(REQUIRED_COMMENT_POST_ID)
    me = client.me()
    my_name = None
    if isinstance(me, dict):
        agent_obj = me.get("agent") if isinstance(me.get("agent"), dict) else None
        my_name = (agent_obj or {}).get("name") or me.get("name")

    # “必做评论”判定：要求在欢迎帖下存在一条英文评论（避免你之前中文评论导致网页看不到“英文新评论”）
    has_english_required_comment = False
    if isinstance(required_post_snapshot, dict) and my_name:
        comments = required_post_snapshot.get("comments") or []
        if isinstance(comments, list):
            for c in comments:
                if not isinstance(c, dict):
                    continue
                author = c.get("author") or {}
                if isinstance(author, dict) and author.get("name") == my_name:
                    content = (c.get("content") or "").strip()
                    if _is_english_comment(content):
                        has_english_required_comment = True
                        break

    if has_english_required_comment:
        if REQUIRED_COMMENT_POST_ID not in state["commented_post_ids"]:
            state["commented_post_ids"].append(REQUIRED_COMMENT_POST_ID)
            _save_state(state_path, state)

    posts = []
    for item in (feed.get("posts") or feed.get("data") or feed.get("items") or []):
        if not isinstance(item, dict):
            continue
        posts.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "content": _truncate(item.get("content") or "", 200),
                "upvotes": item.get("upvotes"),
                "comment_count": item.get("comment_count"),
                "author": (item.get("author") or {}).get("name") if isinstance(item.get("author"), dict) else None,
            }
        )

    feed_post_ids = {p.get("id") for p in posts if isinstance(p, dict) and p.get("id")}

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=llm_api_key, temperature=0.2)
    llm2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=llm_api_key, temperature=0.2)

    system = (
        "You are an autonomous Moltbook agent participating in m/ftec5660.\n"
        "You must be selective and avoid spam.\n"
        "All comments MUST be written in English.\n"
        "Return ONLY valid JSON in the specified schema.\n"
        "Allowed actions: subscribe, upvote, comment, stop.\n"
        "Constraints:\n"
        "- Avoid repetitive/spammy interactions; you MAY comment again if you have something new to add.\n"
        "- Keep comments short, friendly, and relevant to the post.\n"
        "- If you cannot produce an English comment, choose action=stop.\n"
    )

    skill_excerpt = "\n".join(skill_md.splitlines()[:220])

    # 为了让 dry-run 也能走完整决策链：在内存里模拟 state 更新，但不落盘
    sim_state = {
        "subscribed_submolts": list(state.get("subscribed_submolts", [])),
        "upvoted_post_ids": list(state.get("upvoted_post_ids", [])),
        "commented_post_ids": list(state.get("commented_post_ids", [])),
        "last_action_at": state.get("last_action_at"),
    }

    actions_done: list[dict[str, Any]] = []

    for step in range(1, max_actions + 1):
        must_subscribe = TARGET_SUBMOLT not in sim_state["subscribed_submolts"]
        must_upvote = TARGET_POST_ID not in sim_state["upvoted_post_ids"]
        # 必做评论的完成条件：欢迎帖下必须存在“英文”评论。
        # 注意：state 里可能已有旧的中文评论记录，不能据此认为已满足英文要求。
        must_comment_required = not has_english_required_comment
        if must_subscribe:
            allowed_actions = ["subscribe", "stop"]
        elif must_upvote:
            allowed_actions = ["upvote", "stop"]
        elif must_comment_required:
            allowed_actions = ["comment", "stop"]
        else:
            allowed_actions = ["subscribe", "upvote", "comment", "stop"]

        prompt_obj = {
            "skill_md_excerpt": skill_excerpt,
            "submolt": TARGET_SUBMOLT,
            "feed_posts": posts,
            "required_tasks": {
                "subscribe_submolt": TARGET_SUBMOLT,
                "upvote_post_id": TARGET_POST_ID,
                "comment_post_id": REQUIRED_COMMENT_POST_ID,
                "note": "Complete required tasks first (subscribe, upvote, comment on the required welcome post), then optionally engage with other feed posts.",
            },
            "already_subscribed_submolts": sim_state["subscribed_submolts"],
            "already_upvoted_post_ids": sim_state["upvoted_post_ids"],
            "already_commented_post_ids": sim_state["commented_post_ids"],
            "allowed_actions": allowed_actions,
            "time_utc": _iso_utc(),
            "schema": {
                "action": "subscribe|upvote|comment|stop",
                "submolt": "string (required if action=subscribe)",
                "post_id": "string (required if action=upvote/comment)",
                "comment": "string (required if action=comment)",
                "reason": "string (short)",
            },
        }

        msg = system + "\n\nDecide your next action.\n\n" + json.dumps(prompt_obj, ensure_ascii=False)
        try:
            try:
                resp = llm.invoke(msg)
            except Exception:
                resp = llm2.invoke(msg)
            raw = (resp.content or "").strip()
        except Exception as e:
            return {"success": False, "error": "LLM call failed", "detail": str(e)}

        try:
            plan = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            plan = json.loads(m.group(0)) if m else {"action": "stop", "reason": "Invalid output"}

        action = (plan.get("action") or "").strip().lower()
        post_id = (plan.get("post_id") or "").strip()
        submolt = (plan.get("submolt") or "").strip()

        if action not in allowed_actions:
            actions_done.append(
                {"step": step, "action": "stop", "reason": f"Action '{action}' not allowed. Allowed: {allowed_actions}"}
            )
            break

        if action == "stop":
            actions_done.append({"step": step, "action": "stop", "reason": plan.get("reason")})
            break

        if action == "subscribe":
            # 必做阶段：强制订阅目标 submolt（即使 LLM 漏填/填错）
            if must_subscribe:
                submolt = TARGET_SUBMOLT
            if not submolt:
                actions_done.append({"step": step, "action": "stop", "reason": "Missing submolt"})
                break
            if submolt in sim_state["subscribed_submolts"]:
                actions_done.append({"step": step, "action": "skip", "reason": "already subscribed", "submolt": submolt})
                continue

            result = {"success": True, "dry_run": True} if dry_run else client.subscribe(submolt)
            if result.get("success") is True:
                sim_state["subscribed_submolts"].append(submolt)
                sim_state["last_action_at"] = _iso_utc()
                if not dry_run:
                    state["subscribed_submolts"] = sim_state["subscribed_submolts"]
                    state["last_action_at"] = sim_state["last_action_at"]
                    _save_state(state_path, state)
            actions_done.append({"step": step, "action": "subscribe", "submolt": submolt, "result": result, "reason": plan.get("reason")})
            continue

        if action == "upvote":
            # 必做阶段：强制给目标帖点赞（避免 LLM 幻觉 post_id 导致 stop）
            if must_upvote:
                post_id = TARGET_POST_ID
            if not post_id:
                actions_done.append({"step": step, "action": "stop", "reason": "Missing post_id"})
                break
            # 防止模型输出不存在的 post_id
            if post_id != TARGET_POST_ID and post_id not in feed_post_ids:
                actions_done.append({"step": step, "action": "stop", "reason": f"Invalid post_id (not in feed): {post_id}"})
                break
            if post_id in sim_state["upvoted_post_ids"]:
                actions_done.append({"step": step, "action": "skip", "reason": "already upvoted", "post_id": post_id})
                continue

            result = {"success": True, "dry_run": True} if dry_run else client.upvote_post(post_id)
            if result.get("success") is True:
                # upvote 可能 toggle：尽量按 action 字段同步 state
                act = (result.get("action") or "").lower()
                # 如果本次把赞 toggle 掉了，立刻再点一次，尽量在一次 run 内回到 upvoted 状态
                toggled_back = None
                if (not dry_run) and act in {"removed", "unvoted"}:
                    try:
                        toggled_back = client.upvote_post(post_id)
                        act = (toggled_back.get("action") or "").lower() if isinstance(toggled_back, dict) else act
                        result = {"first": result, "second": toggled_back}
                    except Exception:
                        pass

                if dry_run or act in {"upvoted", "success"}:
                    if post_id not in sim_state["upvoted_post_ids"]:
                        sim_state["upvoted_post_ids"].append(post_id)
                elif act in {"removed", "unvoted"}:
                    sim_state["upvoted_post_ids"] = [x for x in sim_state["upvoted_post_ids"] if x != post_id]
                sim_state["last_action_at"] = _iso_utc()
                if not dry_run:
                    state["upvoted_post_ids"] = sim_state["upvoted_post_ids"]
                    state["last_action_at"] = sim_state["last_action_at"]
                    _save_state(state_path, state)
            actions_done.append({"step": step, "action": "upvote", "post_id": post_id, "result": result, "reason": plan.get("reason")})
            continue

        if action == "comment":
            if not post_id:
                # 必做评论阶段允许 LLM 漏填 post_id：直接强制改为欢迎帖
                if must_comment_required:
                    post_id = REQUIRED_COMMENT_POST_ID
                else:
                    actions_done.append({"step": step, "action": "stop", "reason": "Missing post_id"})
                    break
            # 如果必做评论还没完成，只允许评论在指定欢迎帖
            if must_comment_required and post_id != REQUIRED_COMMENT_POST_ID:
                post_id = REQUIRED_COMMENT_POST_ID
            # 非必做评论时，要求 post_id 必须来自 feed sample，避免 404
            if (not must_comment_required) and (post_id not in feed_post_ids) and (post_id != REQUIRED_COMMENT_POST_ID):
                actions_done.append({"step": step, "action": "stop", "reason": f"Invalid post_id (not in feed): {post_id}"})
                break
            comment = (plan.get("comment") or "").strip()
            if not comment:
                # 必做评论缺失时，用固定英文短评兜底（保证完成作业要求）
                if must_comment_required:
                    comment = DEFAULT_REQUIRED_COMMENT_EN
                else:
                    actions_done.append({"step": step, "action": "stop", "reason": "Missing comment"})
                    break
            if _contains_cjk(comment):
                rewritten = _rewrite_to_english(llm=llm, llm2=llm2, text=comment)
                comment = rewritten or "Thanks for sharing—looking forward to learning and discussing this in FTEC5660."

            result = {"success": True, "dry_run": True} if dry_run else client.comment_post(post_id, comment)
            # 若评论需要验证：立即尝试自动验证并发布
            if (not dry_run) and isinstance(result, dict) and result.get("verification_required") is True:
                v = result.get("verification") if isinstance(result.get("verification"), dict) else {}
                vcode = (v.get("code") or "").strip()
                challenge = (v.get("challenge") or "").strip()
                answer = _solve_verification_with_llm(llm=llm, llm2=llm2, challenge=challenge)
                verify_res = None
                if vcode and answer:
                    verify_res = client.verify(vcode, answer)
                result["auto_verify"] = {"answer": answer, "verify_result": verify_res}

                # 若验证没成功，把 code/challenge 记到 state，便于下次重试
                if not (isinstance(verify_res, dict) and verify_res.get("success") is True):
                    state["pending_verifications"][post_id] = {
                        "code": vcode,
                        "challenge": challenge,
                        "answer": answer,
                        "comment_id": (result.get("comment") or {}).get("id") if isinstance(result.get("comment"), dict) else None,
                        "created_at": _iso_utc(),
                    }
                    _save_state(state_path, state)

            # 只有在“评论成功且不需要验证”或“验证成功”时，才把它记为已评论（避免 pending 卡死）
            verified_ok = False
            if isinstance(result, dict) and result.get("verification_required") is True:
                vr = (result.get("auto_verify") or {}).get("verify_result") if isinstance(result.get("auto_verify"), dict) else None
                verified_ok = isinstance(vr, dict) and vr.get("success") is True
            if result.get("success") is True and (result.get("verification_required") is not True or verified_ok):
                if post_id not in sim_state["commented_post_ids"]:
                    sim_state["commented_post_ids"].append(post_id)
                sim_state["last_action_at"] = _iso_utc()
                if not dry_run:
                    state["commented_post_ids"] = sim_state["commented_post_ids"]
                    state["last_action_at"] = sim_state["last_action_at"]
                    # 如果之前有 pending，验证后清掉
                    if isinstance(state.get("pending_verifications"), dict):
                        state["pending_verifications"].pop(post_id, None)
                    _save_state(state_path, state)
            actions_done.append(
                {"step": step, "action": "comment", "post_id": post_id, "comment": comment, "result": result, "reason": plan.get("reason")}
            )
            continue

        actions_done.append({"step": step, "action": "stop", "reason": f"Unknown action: {action}"})
        break

    out: dict[str, Any] = {
        "success": True,
        "read_skill_md_url": SKILL_MD_URL,
        "skill_md_bytes": len(skill_md.encode("utf-8", errors="ignore")),
        "feed_sample_count": len(posts),
        "state_path": str(state_path),
        "actions_done": actions_done,
    }
    if verbose:
        out["feed_sample"] = posts[:5]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Moltbook autonomous agent (agent-only)")
    parser.add_argument("--api-key", default=None, help="可选：直接传 Moltbook API key（不推荐；优先 env/credentials.json）。")
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="可选：传 Gemini API key（优先用 env：VERTEX_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY）。",
    )
    parser.add_argument("--check-only", action="store_true", help="只检查与抓取快照，不执行任何互动。")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH), help="state 文件路径（避免重复互动）。")
    parser.add_argument("--max-actions", type=int, default=3, help="最多决策/执行多少步（默认 3）。")
    parser.add_argument("--dry-run", action="store_true", help="只让 agent 规划动作，不真实调用 subscribe/upvote/comment。")
    parser.add_argument("--verbose-agent", action="store_true", help="输出更多 feed 采样信息。")
    args = parser.parse_args()

    api_key = _load_moltbook_api_key(args.api_key)
    client = MoltbookClient(api_key=api_key)

    results: dict[str, Any] = {}
    results["auth_status"] = client.status()
    results["auth_me"] = client.me()
    results["target_post_snapshot_before"] = client.get_post(TARGET_POST_ID)

    if not args.check_only:
        llm_key = _load_llm_api_key(args.llm_api_key)
        if not llm_key:
            results["autonomous_agent"] = {
                "success": False,
                "error": "Missing LLM API key",
                "hint": "请设置 VERTEX_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY，或传 --llm-api-key。",
            }
        else:
            results["autonomous_agent"] = autonomous_agent_run(
                client=client,
                llm_api_key=llm_key,
                state_path=Path(args.state_path),
                max_actions=max(1, int(args.max_actions)),
                dry_run=bool(args.dry_run),
                verbose=bool(args.verbose_agent),
            )

    results["target_post_snapshot_after"] = client.get_post(TARGET_POST_ID)
    print(_pretty(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

