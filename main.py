#!/usr/bin/env python3
import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import List, Dict, Optional, Any, TypedDict, Literal
from urllib.parse import urlparse


PROJECT_VENV = Path(__file__).with_name("venv312")
PROJECT_VENV_PYTHON = PROJECT_VENV / "bin" / "python"

if (
    os.environ.get("AI_ADVENT_SKIP_VENV") != "1"
    and PROJECT_VENV_PYTHON.exists()
    and Path(sys.executable).resolve() != PROJECT_VENV_PYTHON.resolve()
):
    os.environ["VIRTUAL_ENV"] = str(PROJECT_VENV)
    os.environ["PATH"] = f"{PROJECT_VENV / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    os.execv(str(PROJECT_VENV_PYTHON), [str(PROJECT_VENV_PYTHON), *sys.argv])

import requests
import yaml
from jsonschema import validate, ValidationError
from mcp_client import (
    MCPClientError,
    MCPCodeReviewClient,
    MCPNewsClient,
    MCPPeriodicSummaryClient,
)

DEFAULT_API_BASE = "http://localhost:8080"
DEFAULT_API_PATH = "/v1/chat/completions"


def normalize_api_url(value: str) -> str:
    """Accepts host[:port], base URL, or full chat completions URL."""
    url = value.strip().rstrip("/")
    if not url:
        url = DEFAULT_API_BASE
    if "://" not in url:
        url = f"http://{url}"
    if not url.endswith(DEFAULT_API_PATH):
        url = url.rstrip("/") + DEFAULT_API_PATH
    return url


def api_base_from_url(api_url: str) -> str:
    return api_url.split("/v1/", 1)[0].rstrip("/")


API_URL = normalize_api_url(
    os.environ.get("AI_ADVENT_API_URL")
    or os.environ.get("AI_ADVENT_API_BASE")
    or DEFAULT_API_BASE
)
DEFAULT_TEMPERATURE = 0.7
RAG_CONFIG_FILE = Path(__file__).with_name("rag_indexer_tool") / "config.yaml"
DEFAULT_RAG_SCORE_THRESHOLD = 0.72
DEFAULT_RAG_PRE_TOP_K = 10
DEFAULT_RAG_TOP_K = 5
DEFAULT_RAG_MAX_CONTEXT_CHARS = 12000
DEFAULT_RAG_QUOTE_CHARS = 280
HISTORY_FILE = Path(__file__).with_name("conversation_history.json")
SUMMARY_FILE = Path(__file__).with_name("conversation_summary.json")
FACTS_FILE = Path(__file__).with_name("conversation_facts.json")
BRANCHES_FILE = Path(__file__).with_name("conversation_branches.json")
PROFILE_FILE = Path(__file__).with_name("user_profile.json")
TASK_STATE_FILE = Path(__file__).with_name("task_state.json")
INVARIANTS_FILE = Path(__file__).with_name("invariants.json")
MEMORY_FILE = Path(__file__).with_name("agent_memory.json")
PERIODIC_SUMMARY_FILE = Path(__file__).with_name("periodic_dialog_summary.json")
RECENT_MESSAGES_LIMIT = 10
DEFAULT_SUMMARY_INTERVAL_SECONDS = 300
DEFAULT_SUMMARY_MIN_MESSAGES = 4
TASK_STAGES = ("collecting", "planning", "execution", "validation", "done")
ALLOWED_TASK_TRANSITIONS = {
    "collecting": {"planning"},
    "planning": {"execution"},
    "execution": {"planning", "validation"},
    "validation": {"execution", "done"},
    "done": set(),
}
CONTEXT_STRATEGY_RECENT = "recent"
CONTEXT_STRATEGY_FACTS = "facts"
CONTEXT_STRATEGY_BRANCHES = "branches"
CONTEXT_STRATEGIES = {
    CONTEXT_STRATEGY_RECENT,
    CONTEXT_STRATEGY_FACTS,
    CONTEXT_STRATEGY_BRANCHES,
}


class LLMResponse(TypedDict):
    answer: str
    confidence: float
    intent: Literal["question", "command", "statement"]
    data: Dict[str, Any]


class LLMChat:
    # Определяем схему JSON
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "intent": {"type": "string", "enum": ["question", "command", "statement"]},
            "data": {"type": "object", "additionalProperties": True},
        },
        "required": ["answer", "confidence", "intent", "data"],
    }

    def __init__(
        self,
        api_url: str = API_URL,
        history_file: Path = HISTORY_FILE,
        summary_file: Path = SUMMARY_FILE,
        facts_file: Path = FACTS_FILE,
        branches_file: Path = BRANCHES_FILE,
        profile_file: Path = PROFILE_FILE,
        task_state_file: Path = TASK_STATE_FILE,
        invariants_file: Path = INVARIANTS_FILE,
        memory_file: Optional[Path] = None,
        periodic_summary_file: Path = PERIODIC_SUMMARY_FILE,
        context_strategy: str = CONTEXT_STRATEGY_RECENT,
    ):
        self.api_url: str = api_url
        self.api_base_url: str = api_base_from_url(api_url)
        self.tokenize_url: str = self.api_base_url + "/tokenize"
        self.history_file = history_file
        self.summary_file = summary_file
        self.facts_file = facts_file
        self.branches_file = branches_file
        self.profile_file = profile_file
        self.task_state_file = task_state_file
        self.invariants_file = invariants_file
        self.memory_file = memory_file or (
            MEMORY_FILE
            if profile_file == PROFILE_FILE
            else profile_file.with_name("agent_memory.json")
        )
        self.periodic_summary_file = periodic_summary_file
        self.context_strategy = (
            context_strategy
            if context_strategy in CONTEXT_STRATEGIES
            else CONTEXT_STRATEGY_RECENT
        )
        self.conversation_history: List[Dict[str, str]] = self.load_history()
        self.conversation_summary: str = self.load_summary()
        self.facts: Dict[str, str] = self.load_facts()
        branch_state = self.load_branches()
        self.branches: Dict[str, List[Dict[str, str]]] = branch_state["branches"]
        self.current_branch: Optional[str] = branch_state["current_branch"]
        profile_state = self.load_profiles()
        self.profiles: Dict[str, Dict[str, str]] = profile_state["profiles"]
        self.current_user: str = profile_state["current_user"]
        self.profile: Dict[str, str] = self.profiles.setdefault(
            self.current_user, {}
        )
        self.task_state: Dict[str, Any] = self.load_task_state()
        self.invariants: List[Dict[str, Any]] = self.load_invariants()
        legacy_memory = self.create_memory(
            history=self.conversation_history,
            summary=self.conversation_summary,
            branches=self.branches,
            current_branch=self.current_branch,
            task=self.task_state,
            knowledge=self.facts,
            invariants=self.invariants,
        )
        self.memories: Dict[str, Dict[str, Any]] = self.load_memories(legacy_memory)
        self.activate_user_memory(self.current_user, sync_current=False)
        self.normalize_invariants()
        self.running: bool = True
        self.tokenize_available: Optional[bool] = None
        self.last_token_counts: Optional[Dict[str, int]] = None
        self.mcp_client = MCPNewsClient()
        self.summary_mcp_client = MCPPeriodicSummaryClient()
        self.code_review_mcp_client = MCPCodeReviewClient()
        self.last_code_review_problems: List[Dict[str, Any]] = []
        self.last_code_review_folder: Optional[str] = None
        self.summary_stop_event = Event()
        self.summary_thread: Optional[Thread] = None
        self.summary_last_message_count = 0
        self.summary_interval_seconds = self.get_env_int(
            "AI_ADVENT_SUMMARY_INTERVAL_SECONDS",
            DEFAULT_SUMMARY_INTERVAL_SECONDS,
        )
        self.summary_min_messages = self.get_env_int(
            "AI_ADVENT_SUMMARY_MIN_MESSAGES",
            DEFAULT_SUMMARY_MIN_MESSAGES,
        )
        self.rag_config = self.load_rag_config()
        self.rag_enabled = self.get_env_bool("AI_ADVENT_RAG_ENABLED", True)
        self.rag_score_threshold = self.get_env_float(
            "AI_ADVENT_RAG_SCORE_THRESHOLD",
            DEFAULT_RAG_SCORE_THRESHOLD,
        )
        self.rag_pre_top_k = self.get_env_int(
            "AI_ADVENT_RAG_PRE_TOP_K", DEFAULT_RAG_PRE_TOP_K
        )
        self.rag_top_k = self.get_env_int("AI_ADVENT_RAG_TOP_K", DEFAULT_RAG_TOP_K)
        self.rag_max_context_chars = self.get_env_int(
            "AI_ADVENT_RAG_MAX_CONTEXT_CHARS",
            DEFAULT_RAG_MAX_CONTEXT_CHARS,
        )
        self.rag_quote_chars = self.get_env_int(
            "AI_ADVENT_RAG_QUOTE_CHARS", DEFAULT_RAG_QUOTE_CHARS
        )
        self.rag_query_rewrite_enabled = self.get_env_bool(
            "AI_ADVENT_RAG_QUERY_REWRITE", True
        )
        self.rag_strict_unknown = self.get_env_bool(
            "AI_ADVENT_RAG_STRICT_UNKNOWN", True
        )
        self.trim_active_history_if_needed()

    def create_memory(
        self,
        history: Optional[List[Dict[str, str]]] = None,
        summary: str = "",
        branches: Optional[Dict[str, List[Dict[str, str]]]] = None,
        current_branch: Optional[str] = None,
        task: Optional[Dict[str, Any]] = None,
        knowledge: Optional[Dict[str, str]] = None,
        invariants: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "short_term": {
                "history": history or [],
                "summary": summary,
                "branches": branches or {},
                "current_branch": current_branch,
            },
            "working": {
                "task": task or {"description": "", "stage": None, "plan": ""},
                "notes": {},
            },
            "long_term": {
                "decisions": {},
                "knowledge": knowledge or {},
                "invariants": invariants or [],
            },
        }

    def get_env_int(self, name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            print(f"{name} должен быть числом, используется {default}")
            return default

    def get_env_float(self, name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            print(f"{name} должен быть числом, используется {default}")
            return default

    def get_env_bool(self, name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().casefold() not in {"0", "false", "no", "off", "нет"}

    def load_rag_config(self) -> Dict[str, Any]:
        if not RAG_CONFIG_FILE.exists():
            return {}
        try:
            with RAG_CONFIG_FILE.open("r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, yaml.YAMLError) as error:
            print(f"Не удалось загрузить RAG config: {error}")
            return {}

    def load_memories(self, legacy_memory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not self.memory_file.exists():
            return {self.current_user: legacy_memory}
        try:
            with self.memory_file.open("r", encoding="utf-8") as file:
                value = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить память пользователей: {e}")
            return {self.current_user: legacy_memory}
        if not isinstance(value, dict):
            print("Файл памяти пользователей имеет неверный формат")
            return {self.current_user: legacy_memory}

        memories: Dict[str, Dict[str, Any]] = {}
        for user_id, memory in value.items():
            if not isinstance(user_id, str) or not isinstance(memory, dict):
                continue
            empty = self.create_memory()
            for layer in ("short_term", "working", "long_term"):
                stored_layer = memory.get(layer)
                if isinstance(stored_layer, dict):
                    empty[layer].update(stored_layer)
            memories[user_id] = empty
        memories.setdefault(self.current_user, legacy_memory)
        return memories

    def sync_active_memory(self) -> None:
        if not hasattr(self, "memories"):
            return
        self.memories[self.current_user] = {
            "short_term": {
                "history": self.conversation_history,
                "summary": self.conversation_summary,
                "branches": self.branches,
                "current_branch": self.current_branch,
            },
            "working": {"task": self.task_state, "notes": self.working_notes},
            "long_term": {
                "decisions": self.decisions,
                "knowledge": self.facts,
                "invariants": self.invariants,
            },
        }

    def activate_user_memory(self, user_id: str, sync_current: bool = True) -> None:
        if sync_current:
            self.sync_active_memory()
        memory = self.memories.setdefault(user_id, self.create_memory())
        short_term = memory["short_term"]
        working = memory["working"]
        long_term = memory["long_term"]
        self.conversation_history = short_term["history"]
        self.conversation_summary = str(short_term.get("summary", ""))
        self.branches = short_term["branches"]
        self.current_branch = short_term.get("current_branch")
        self.task_state = working["task"]
        self.working_notes: Dict[str, str] = working["notes"]
        self.decisions: Dict[str, str] = long_term["decisions"]
        self.facts = long_term["knowledge"]
        self.invariants = long_term["invariants"]

    def save_memories(self) -> None:
        self.sync_active_memory()
        self.save_json_file(self.memory_file, self.memories, "память пользователей")

    def load_profiles(self) -> Dict[str, Any]:
        empty = {"current_user": "default", "profiles": {"default": {}}}
        if not self.profile_file.exists():
            return empty
        try:
            with self.profile_file.open("r", encoding="utf-8") as file:
                value = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить профили: {e}")
            return empty
        if not isinstance(value, dict):
            print("Файл профилей имеет неверный формат")
            return empty

        # Миграция старого формата: {"style": "...", "context": "..."}.
        if "profiles" not in value:
            legacy_profile = {
                str(key): str(item) for key, item in value.items()
            }
            return {
                "current_user": "default",
                "profiles": {"default": legacy_profile},
            }

        raw_profiles = value.get("profiles")
        if not isinstance(raw_profiles, dict):
            print("Файл профилей имеет неверный формат")
            return empty
        profiles: Dict[str, Dict[str, str]] = {}
        for user_id, profile in raw_profiles.items():
            if not isinstance(user_id, str) or not isinstance(profile, dict):
                continue
            profiles[user_id] = {
                str(key): str(item) for key, item in profile.items()
            }
        if not profiles:
            profiles = {"default": {}}
        current_user = value.get("current_user")
        if not isinstance(current_user, str) or current_user not in profiles:
            current_user = next(iter(profiles))
        return {"current_user": current_user, "profiles": profiles}

    def load_task_state(self) -> Dict[str, Any]:
        empty = {"description": "", "stage": None, "plan": ""}
        if not self.task_state_file.exists():
            return empty
        try:
            with self.task_state_file.open("r", encoding="utf-8") as file:
                state = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить состояние задачи: {e}")
            return empty
        if not isinstance(state, dict) or state.get("stage") not in (*TASK_STAGES, None):
            print("Файл состояния задачи имеет неверный формат")
            return empty
        return {
            "description": str(state.get("description", "")),
            "stage": state.get("stage"),
            "plan": str(state.get("plan", "")),
        }

    def load_invariants(self) -> List[Dict[str, Any]]:
        if not self.invariants_file.exists():
            return []
        try:
            with self.invariants_file.open("r", encoding="utf-8") as file:
                values = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить инварианты: {e}")
            return []
        if not isinstance(values, list):
            print("Файл инвариантов имеет неверный формат")
            return []
        result = []
        for value in values:
            if not isinstance(value, dict) or not str(value.get("rule", "")).strip():
                continue
            forbidden = value.get("forbidden_terms", [])
            result.append({
                "rule": str(value["rule"]).strip(),
                "forbidden_terms": [str(term).strip() for term in forbidden]
                if isinstance(forbidden, list) else [],
            })
        return result

    def load_history(self) -> List[Dict[str, str]]:
        """Загружает историю диалога из файла между запусками."""
        if not self.history_file.exists():
            return []

        try:
            with self.history_file.open("r", encoding="utf-8") as file:
                history = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить историю диалога: {e}")
            return []

        if not isinstance(history, list):
            print("Файл истории имеет неверный формат, начинаем с пустой истории")
            return []

        valid_history: List[Dict[str, str]] = []
        for message in history:
            if (
                isinstance(message, dict)
                and message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
            ):
                valid_history.append(
                    {"role": message["role"], "content": message["content"]}
                )

        skipped = len(history) - len(valid_history)
        if skipped:
            print(f"Пропущено некорректных сообщений в истории: {skipped}")

        return valid_history

    def load_summary(self) -> str:
        """Загружает summary старой части диалога из отдельного файла."""
        if not self.summary_file.exists():
            return ""

        try:
            with self.summary_file.open("r", encoding="utf-8") as file:
                summary_data = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить summary диалога: {e}")
            return ""

        if isinstance(summary_data, dict) and isinstance(summary_data.get("summary"), str):
            return summary_data["summary"]
        if isinstance(summary_data, str):
            return summary_data

        print("Файл summary имеет неверный формат, начинаем без summary")
        return ""

    def load_facts(self) -> Dict[str, str]:
        """Загружает facts из файла."""
        if not self.facts_file.exists():
            return {}

        try:
            with self.facts_file.open("r", encoding="utf-8") as file:
                facts = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить facts: {e}")
            return {}

        if not isinstance(facts, dict):
            print("Файл facts имеет неверный формат, начинаем без facts")
            return {}

        return {str(key): str(value) for key, value in facts.items()}

    def load_branches(self) -> Dict[str, Any]:
        """Загружает состояние веток диалога."""
        empty_state = {"current_branch": None, "branches": {}}
        if not self.branches_file.exists():
            return empty_state

        try:
            with self.branches_file.open("r", encoding="utf-8") as file:
                state = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить ветки диалога: {e}")
            return empty_state

        if not isinstance(state, dict) or not isinstance(state.get("branches"), dict):
            print("Файл веток имеет неверный формат, начинаем без веток")
            return empty_state

        branches: Dict[str, List[Dict[str, str]]] = {}
        for name, history in state["branches"].items():
            if not isinstance(name, str) or not isinstance(history, list):
                continue
            valid_history: List[Dict[str, str]] = []
            for message in history:
                if (
                    isinstance(message, dict)
                    and message.get("role") in {"user", "assistant"}
                    and isinstance(message.get("content"), str)
                ):
                    valid_history.append(
                        {"role": message["role"], "content": message["content"]}
                    )
            branches[name] = valid_history

        current_branch = state.get("current_branch")
        if not isinstance(current_branch, str) or current_branch not in branches:
            current_branch = next(iter(branches), None)

        return {"current_branch": current_branch, "branches": branches}

    def save_history(self) -> None:
        """Сохраняет краткосрочную память активного пользователя."""
        self.save_memories()

    def save_summary(self) -> None:
        self.save_memories()

    def save_facts(self) -> None:
        """Сохраняет знания в долговременную память активного пользователя."""
        self.save_memories()

    def save_branches(self) -> None:
        self.save_memories()

    def save_json_file(self, path: Path, value: Any, label: str) -> None:
        try:
            with path.open("w", encoding="utf-8") as file:
                json.dump(value, file, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"Не удалось сохранить {label}: {e}")

    def save_profile(self) -> None:
        self.save_json_file(
            self.profile_file,
            {"current_user": self.current_user, "profiles": self.profiles},
            "профили",
        )

    def save_task_state(self) -> None:
        self.save_memories()

    def save_invariants(self) -> None:
        self.save_memories()

    def save_conversation_state(self) -> None:
        self.save_memories()
        self.save_profile()

    def build_stateful_system_prompt(self, base_prompt: str) -> str:
        """Собирает только актуальные для текущей задачи слои контекста."""
        blocks = [base_prompt.strip()]
        if self.profile:
            blocks.append(
                f"ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ {self.current_user!r} "
                "(учитывай стиль, ограничения и контекст):\n"
                + json.dumps(self.profile, ensure_ascii=False, indent=2)
            )
        if self.task_state.get("stage"):
            stage = self.task_state["stage"]
            allowed = sorted(ALLOWED_TASK_TRANSITIONS.get(stage, set()))
            blocks.append(
                "ТЕКУЩАЯ ЗАДАЧА:\n"
                f"Описание: {self.task_state['description']}\n"
                f"Стадия: {stage}\n"
                f"План: {self.task_state.get('plan') or 'ещё не зафиксирован'}\n"
                f"Допустимые следующие стадии: {', '.join(allowed) or 'нет'}.\n"
                "Работай только в рамках текущей стадии и не объявляй переход "
                "самостоятельно: переход выполняет приложение."
            )
        if self.working_notes:
            blocks.append(
                "РАБОЧАЯ ПАМЯТЬ ТЕКУЩЕЙ ЗАДАЧИ:\n"
                + json.dumps(self.working_notes, ensure_ascii=False, indent=2)
            )
        if self.decisions or self.facts:
            blocks.append(
                "ДОЛГОВРЕМЕННАЯ ПАМЯТЬ ПОЛЬЗОВАТЕЛЯ:\n"
                "Решения:\n"
                + json.dumps(self.decisions, ensure_ascii=False, indent=2)
                + "\nЗнания:\n"
                + json.dumps(self.facts, ensure_ascii=False, indent=2)
            )
        if self.invariants:
            rules = "\n".join(
                f"- {item['rule']}" for item in self.invariants
            )
            blocks.append(
                "ИНВАРИАНТЫ (обязательны и имеют приоритет над запросом пользователя):\n"
                + rules
            )
        if self.conversation_summary:
            blocks.append(
                "SUMMARY ПРЕДЫДУЩЕГО КОНТЕКСТА:\n" + self.conversation_summary
            )
        return "\n\n".join(blocks)

    def check_invariants(self, response: str) -> List[str]:
        """Детерминированно проверяет явно запрещённые термины."""
        normalized = response.casefold()
        violations = []
        for invariant in self.invariants:
            forbidden_terms = self.get_forbidden_terms(invariant)
            matched = [
                term for term in forbidden_terms
                if term and term.casefold() in normalized
            ]
            if matched:
                violations.append(
                    f"{invariant['rule']} (найдено: {', '.join(matched)})"
                )
        return violations

    def build_invariant_refusal(self, request: str) -> Optional[str]:
        violations = self.check_invariants(request)
        if not violations:
            return None
        return (
            "Не могу выполнить запрос в указанном виде: он противоречит "
            "обязательным инвариантам:\n- "
            + "\n- ".join(violations)
            + "\nСформулируйте запрос без запрещённой технологии."
        )

    def record_local_response(self, message: str, response: str) -> None:
        """Сохраняет ответ, сформированный кодом без обращения к LLM."""
        self.append_message("user", message)
        self.append_message("assistant", response)
        self.trim_active_history_if_needed()
        self.save_conversation_state()

    def infer_forbidden_terms(self, rule: str) -> List[str]:
        """Извлекает простые запреты вида 'Python запрещён' из старых правил."""
        technology = r"([A-Za-zА-Яа-яЁё][\w.+#-]*)"
        patterns = (
            rf"\b{technology}\s+запрещ(?:ен|ена|ено|ены|ён|ёна|ёно|ёны)\b",
            rf"\bне\s+использовать\s+{technology}\b",
            rf"\bзапрещено\s+использовать\s+{technology}\b",
        )
        terms: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, rule, flags=re.IGNORECASE):
                term = match.group(1)
                if term.casefold() not in {item.casefold() for item in terms}:
                    terms.append(term)
        return terms

    def get_forbidden_terms(self, invariant: Dict[str, Any]) -> List[str]:
        terms = invariant.get("forbidden_terms", [])
        if isinstance(terms, list) and terms:
            return [str(term).strip() for term in terms if str(term).strip()]
        return self.infer_forbidden_terms(str(invariant.get("rule", "")))

    def normalize_invariants(self) -> None:
        """Мигрирует старые текстовые запреты в проверяемую структуру."""
        for invariant in self.invariants:
            if not invariant.get("forbidden_terms"):
                invariant["forbidden_terms"] = self.infer_forbidden_terms(
                    str(invariant.get("rule", ""))
                )

    def transition_task(self, target_stage: str) -> bool:
        current = self.task_state.get("stage")
        if current is None:
            print("Сначала создайте задачу: /task <описание>")
            return False
        if target_stage not in TASK_STAGES:
            print(f"Неизвестная стадия. Доступны: {', '.join(TASK_STAGES)}")
            return False
        if target_stage not in ALLOWED_TASK_TRANSITIONS[current]:
            allowed = ", ".join(sorted(ALLOWED_TASK_TRANSITIONS[current])) or "нет"
            print(f"Переход {current} -> {target_stage} запрещён. Допустимо: {allowed}")
            return False
        if current == "planning" and target_stage == "execution" and not self.task_state.get("plan"):
            print("Перед execution зафиксируйте план командой /plan <текст>")
            return False
        self.task_state["stage"] = target_stage
        self.save_task_state()
        print(f"Стадия задачи: {target_stage}")
        return True

    def get_active_history(self) -> List[Dict[str, str]]:
        if (
            self.context_strategy == CONTEXT_STRATEGY_BRANCHES
            and self.current_branch
            and self.current_branch in self.branches
        ):
            return self.branches[self.current_branch]
        return self.conversation_history

    def append_message(self, role: str, content: str) -> None:
        self.get_active_history().append({"role": role, "content": content})

    def trim_active_history_if_needed(self) -> None:
        if self.context_strategy not in {CONTEXT_STRATEGY_RECENT, CONTEXT_STRATEGY_FACTS}:
            return
        active_history = self.get_active_history()
        if len(active_history) > RECENT_MESSAGES_LIMIT:
            del active_history[:-RECENT_MESSAGES_LIMIT]

    def get_summary_source_messages(self) -> List[Dict[str, str]]:
        return [dict(message) for message in self.get_active_history()]

    def load_periodic_summary_text(self) -> str:
        if not self.periodic_summary_file.exists():
            return ""
        try:
            with self.periodic_summary_file.open("r", encoding="utf-8") as file:
                value = json.load(file)
        except (OSError, json.JSONDecodeError):
            return ""
        if isinstance(value, dict) and isinstance(value.get("summary"), str):
            return value["summary"]
        return ""

    def run_periodic_summary_once(self, force: bool = False) -> bool:
        messages = self.get_summary_source_messages()
        message_count = len(messages)
        min_messages = 1 if force else self.summary_min_messages
        if message_count < min_messages:
            if force:
                print("\nНет сообщений для summary\n")
            return False
        if not force and message_count == self.summary_last_message_count:
            return False

        try:
            if not self.summary_mcp_client.is_running:
                self.summary_mcp_client.start()
            result = self.summary_mcp_client.call_tool(
                "summarize_dialog",
                {
                    "api_url": self.api_url,
                    "messages": messages,
                    "previous_summary": self.load_periodic_summary_text(),
                    "output_file": str(self.periodic_summary_file),
                    "current_user": self.current_user,
                    "context_strategy": self.context_strategy,
                },
            )
        except (MCPClientError, OSError) as error:
            if force:
                print(f"\nНе удалось сделать summary: {error}\n")
            return False

        if force:
            for content in result.get("content", []):
                if content.get("type") == "text":
                    print(f"\n{content.get('text', '')}\n")
        if result.get("isError", False):
            return False
        self.summary_last_message_count = message_count
        return True

    def start_periodic_summary(self) -> None:
        if self.summary_interval_seconds <= 0:
            return
        if self.summary_thread and self.summary_thread.is_alive():
            return
        try:
            result = self.summary_mcp_client.start()
            server = result.get("serverInfo", {})
            print(
                "Auto-summary MCP: "
                f"{server.get('name', 'unknown')} {server.get('version', '')}; "
                f"интервал {self.summary_interval_seconds} сек; "
                f"файл {self.periodic_summary_file}"
            )
        except (MCPClientError, OSError) as error:
            print(f"Auto-summary MCP не запущен: {error}")
            return

        self.summary_stop_event.clear()
        self.summary_thread = Thread(
            target=self.periodic_summary_loop,
            name="periodic-summary-mcp",
            daemon=True,
        )
        self.summary_thread.start()

    def periodic_summary_loop(self) -> None:
        while not self.summary_stop_event.wait(self.summary_interval_seconds):
            self.run_periodic_summary_once(force=False)

    def stop_periodic_summary(self) -> None:
        self.summary_stop_event.set()
        if self.summary_thread and self.summary_thread.is_alive():
            self.summary_thread.join(timeout=2)
        self.summary_mcp_client.stop()

    def show_periodic_summary_status(self) -> None:
        print("\nAUTO-SUMMARY:")
        print(f"Файл: {self.periodic_summary_file}")
        print(f"Интервал: {self.summary_interval_seconds} сек")
        print(f"Минимум сообщений: {self.summary_min_messages}")
        print(f"MCP запущен: {'да' if self.summary_mcp_client.is_running else 'нет'}")
        print(f"Файл существует: {'да' if self.periodic_summary_file.exists() else 'нет'}")
        print()

    def count_text_tokens(self, text: str) -> int:
        """Считает токены через локальный сервер, при недоступности использует оценку."""
        if not text:
            return 0

        if self.tokenize_available is not False:
            try:
                response = requests.post(
                    self.tokenize_url,
                    json={"content": text},
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                tokens = result.get("tokens")
                if isinstance(tokens, list):
                    self.tokenize_available = True
                    return len(tokens)
            except Exception:
                self.tokenize_available = False

        return max(1, len(text) // 3)

    def count_history_tokens(self) -> int:
        """Считает токены контекста, который отправляется модели."""
        active_history = self.get_active_history()
        messages_for_context = (
            active_history
            if self.context_strategy == CONTEXT_STRATEGY_BRANCHES
            else active_history[-RECENT_MESSAGES_LIMIT:]
        )
        history_text = self.format_messages(messages_for_context)
        facts_text = json.dumps(self.facts, ensure_ascii=False) if self.facts else ""
        full_context = "\n".join(part for part in [facts_text, history_text] if part)
        return self.count_text_tokens(full_context)

    def build_token_counts(self, current_request: str, model_response: str) -> Dict[str, int]:
        return {
            "current_request": self.count_text_tokens(current_request),
            "history": self.count_history_tokens(),
            "model_response": self.count_text_tokens(model_response),
        }

    def print_token_counts(self, counts: Dict[str, int]) -> None:
        print("Токены:")
        print(f"   Текущий запрос: {counts['current_request']}")
        print(f"   Контекст диалога: {counts['history']}")
        print(f"   Ответ модели: {counts['model_response']}")

    def build_context_messages(self) -> List[Dict[str, str]]:
        """Возвращает контекст согласно выбранной стратегии памяти."""
        context_messages: List[Dict[str, str]] = []
        if self.context_strategy == CONTEXT_STRATEGY_BRANCHES:
            context_messages.extend(self.get_active_history())
        else:
            context_messages.extend(self.get_active_history()[-RECENT_MESSAGES_LIMIT:])
        return context_messages

    def get_rag_setting(self, section: str, key: str, default: str) -> str:
        section_value = self.rag_config.get(section, {})
        if isinstance(section_value, dict):
            value = section_value.get(key, default)
            return str(value)
        return default

    def embed_rag_query(self, message: str) -> List[float]:
        ollama_url = os.environ.get("AI_ADVENT_RAG_OLLAMA_URL") or self.get_rag_setting(
            "embedding", "ollama_url", "http://localhost:11434"
        )
        model = os.environ.get("AI_ADVENT_RAG_EMBEDDING_MODEL") or self.get_rag_setting(
            "embedding", "embedding_model", "nomic-embed-text"
        )
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/embed",
            json={"model": model, "input": [message]},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None and "embedding" in data:
            embeddings = [data["embedding"]]
        if not embeddings or not isinstance(embeddings[0], list):
            raise RuntimeError(f"Unexpected Ollama embedding response: {data.keys()}")
        return embeddings[0]

    def rewrite_rag_query(self, message: str) -> str:
        if not self.rag_query_rewrite_enabled:
            return message
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's question into a concise search query "
                        "for a local vector knowledge base. Preserve exact names, "
                        "codes, product identifiers, numbers, and quoted phrases. "
                        "Return only the rewritten query, no markdown."
                    ),
                },
                {"role": "user", "content": message},
            ],
            "max_tokens": 120,
            "temperature": 0.0,
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=20)
            response.raise_for_status()
            rewritten = response.json()["choices"][0]["message"]["content"].strip()
            return rewritten or message
        except Exception as error:
            if self.get_env_bool("AI_ADVENT_RAG_DEBUG", False):
                print(f"RAG query rewrite недоступен, используется исходный вопрос: {error}")
            return message

    def tokenize_for_rerank(self, text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_.:-]+", text)
            if len(token) > 2
        }

    def rerank_rag_hits(
        self, hits: List[Dict[str, Any]], original_query: str, rewritten_query: str
    ) -> List[Dict[str, Any]]:
        query_tokens = self.tokenize_for_rerank(f"{original_query} {rewritten_query}")
        filtered = [
            dict(hit)
            for hit in hits
            if float(hit.get("score", 0.0) or 0.0) >= self.rag_score_threshold
        ]
        for hit in filtered:
            text_tokens = self.tokenize_for_rerank(
                f"{hit.get('source_path', '')} {hit.get('text', '')}"
            )
            overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
            hit["rerank_score"] = (0.85 * float(hit["score"])) + (0.15 * overlap)
            hit["lexical_overlap"] = overlap
        filtered.sort(key=lambda item: item["rerank_score"], reverse=True)
        return filtered[: self.rag_top_k]

    def query_qdrant_hits(
        self, qdrant_url: str, collection: str, query: str
    ) -> List[Dict[str, Any]]:
        from qdrant_client import QdrantClient

        query_vector = self.embed_rag_query(query)
        client = QdrantClient(url=qdrant_url)
        limit = max(self.rag_pre_top_k, self.rag_top_k)
        if hasattr(client, "query_points"):
            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            raw_points = getattr(result, "points", result)
        else:
            raw_points = client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

        hits: List[Dict[str, Any]] = []
        for point in raw_points:
            score = float(getattr(point, "score", 0.0) or 0.0)
            payload = getattr(point, "payload", None) or {}
            if not isinstance(payload, dict):
                continue
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            hits.append(
                {
                    "score": score,
                    "text": text,
                    "source_path": str(
                        payload.get("source_path")
                        or payload.get("file_name")
                        or "unknown"
                    ),
                    "chunk_index": payload.get("chunk_index"),
                    "heading_path": payload.get("heading_path") or [],
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                }
            )
        return hits

    def search_rag(self, message: str) -> Optional[Dict[str, Any]]:
        if not self.rag_enabled:
            return None
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return None

        qdrant_url = os.environ.get("AI_ADVENT_RAG_QDRANT_URL") or self.get_rag_setting(
            "qdrant", "url", "http://localhost:6333"
        )
        collection = os.environ.get("AI_ADVENT_RAG_COLLECTION") or self.get_rag_setting(
            "qdrant", "collection_name", "local_knowledge_base"
        )

        try:
            rewritten_query = self.rewrite_rag_query(message)
            hits = self.query_qdrant_hits(qdrant_url, collection, rewritten_query)
        except Exception as error:
            if self.get_env_bool("AI_ADVENT_RAG_DEBUG", False):
                print(f"RAG недоступен, используется обычная LLM: {error}")
            return None

        best_score = max((hit["score"] for hit in hits), default=0.0)
        filtered_hits = self.rerank_rag_hits(hits, message, rewritten_query)
        fallback_used = False
        if not filtered_hits and rewritten_query != message:
            try:
                hits = self.query_qdrant_hits(qdrant_url, collection, message)
                best_score = max((hit["score"] for hit in hits), default=0.0)
                filtered_hits = self.rerank_rag_hits(hits, message, message)
                fallback_used = bool(filtered_hits)
            except Exception as error:
                if self.get_env_bool("AI_ADVENT_RAG_DEBUG", False):
                    print(f"RAG fallback по исходному вопросу не сработал: {error}")
        if not filtered_hits:
            return {
                "best_score": best_score,
                "hits": [],
                "raw_hit_count": len(hits),
                "filtered_hit_count": 0,
                "query": message,
                "rewritten_query": rewritten_query,
                "fallback_to_original_query": fallback_used,
                "weak_context": True,
            }
        return {
            "best_score": best_score,
            "hits": filtered_hits,
            "raw_hit_count": len(hits),
            "filtered_hit_count": len(filtered_hits),
            "query": message,
            "rewritten_query": rewritten_query,
            "fallback_to_original_query": fallback_used,
            "weak_context": False,
        }

    def format_rag_context(self, rag_context: Dict[str, Any]) -> str:
        parts: List[str] = []
        used_chars = 0
        for index, hit in enumerate(rag_context["hits"], 1):
            heading = hit["heading_path"]
            heading_text = (
                " > ".join(str(item) for item in heading)
                if isinstance(heading, list)
                else str(heading)
            )
            location_parts = [hit["source_path"]]
            if hit["chunk_index"] is not None:
                location_parts.append(f"chunk {hit['chunk_index']}")
            if hit["page_start"]:
                location_parts.append(f"page {hit['page_start']}")
            if heading_text:
                location_parts.append(heading_text)
            text = hit["text"]
            remaining = self.rag_max_context_chars - used_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip()
            block = (
                f"[{index}] score={hit['score']:.3f}; "
                f"rerank={hit.get('rerank_score', hit['score']):.3f}; "
                f"источник: {', '.join(location_parts)}\n"
                f"{text}"
            )
            parts.append(block)
            used_chars += len(text)
        return "\n\n".join(parts)

    def format_rag_sources(self, rag_context: Dict[str, Any]) -> str:
        lines = []
        seen = set()
        for index, hit in enumerate(rag_context["hits"], 1):
            key = (hit["source_path"], hit.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            chunk = (
                f", chunk {hit['chunk_index']}"
                if hit.get("chunk_index") is not None
                else ""
            )
            lines.append(
                f"{index}. {hit['source_path']}{chunk} "
                f"(score {hit['score']:.3f}, "
                f"rerank {hit.get('rerank_score', hit['score']):.3f})"
            )
        return "\n".join(lines)

    def quote_from_hit(self, hit: Dict[str, Any]) -> str:
        text = " ".join(str(hit.get("text", "")).split())
        if len(text) <= self.rag_quote_chars:
            return text
        return text[: self.rag_quote_chars].rstrip() + "..."

    def format_rag_required_sections(self, answer: str, rag_context: Dict[str, Any]) -> str:
        sources = []
        quotes = []
        for index, hit in enumerate(rag_context["hits"], 1):
            section = hit.get("heading_path") or []
            section_text = (
                " > ".join(str(item) for item in section)
                if isinstance(section, list)
                else str(section)
            )
            chunk_id = hit.get("chunk_index")
            sources.append(
                f"{index}. source={hit['source_path']}; "
                f"section={section_text or 'unknown'}; chunk_id={chunk_id}; "
                f"score={hit['score']:.3f}; "
                f"rerank={hit.get('rerank_score', hit['score']):.3f}"
            )
            quotes.append(f"{index}. \"{self.quote_from_hit(hit)}\"")
        return (
            f"Ответ:\n{answer.strip()}\n\n"
            "Источники:\n"
            + "\n".join(sources)
            + "\n\nЦитаты:\n"
            + "\n".join(quotes)
        )

    def build_weak_context_response(self, rag_context: Dict[str, Any]) -> str:
        return (
            "Ответ:\n"
            "Не знаю: в локальной базе не найден достаточно релевантный контекст. "
            "Пожалуйста, уточните вопрос или добавьте более подходящие документы в RAG.\n\n"
            "Источники:\nнет источников выше порога релевантности\n\n"
            "Цитаты:\nнет цитат выше порога релевантности\n\n"
            f"Диагностика: best_score={rag_context.get('best_score', 0.0):.3f}, "
            f"threshold={self.rag_score_threshold:.3f}"
        )

    def build_rag_system_prompt(
        self, base_prompt: str, rag_context: Optional[Dict[str, Any]]
    ) -> str:
        if not rag_context or rag_context.get("weak_context"):
            return base_prompt
        return (
            f"{base_prompt}\n\n"
            "RAG-КОНТЕКСТ ЛОКАЛЬНОЙ БАЗЫ ЗНАНИЙ:\n"
            f"{self.format_rag_context(rag_context)}\n\n"
            "Если RAG-контекст отвечает на вопрос, отвечай по нему. "
            "Не придумывай факты вне найденных фрагментов; если фрагментов "
            "не хватает, явно скажи, чего не хватает.\n\n"
            "ФОРМАТ ОТВЕТА ОБЯЗАТЕЛЕН:\n"
            "Ответ:\n<краткий ответ по контексту>\n\n"
            "Источники:\n"
            "<нумерованный список: source=<файл>; section=<раздел>; chunk_id=<id>>\n\n"
            "Цитаты:\n"
            "<нумерованный список коротких дословных фрагментов из найденных чанков, "
            "которые подтверждают ответ>"
        )

    def compact_history_if_needed(self) -> None:
        """Оставлено для совместимости со старым кодом: новые стратегии не сжимают history."""
        return

    def update_facts_from_user_message(self, message: str) -> None:
        """Обновляет facts после каждого сообщения пользователя."""
        if self.context_strategy != CONTEXT_STRATEGY_FACTS:
            return

        schema = {
            "type": "object",
            "additionalProperties": True,
        }
        system_prompt = (
            "Ты ведешь компактную долговременную память диалога в виде facts: "
            "JSON object key-value. Обновляй только важные устойчивые данные: цель, "
            "ограничения, предпочтения, решения, договоренности, имена, параметры, "
            "открытые задачи и другие сведения, которые пригодятся позже. "
            "Удаляй или заменяй устаревшие значения. Верни только JSON object."
        )
        user_prompt = (
            f"Текущие facts:\n{json.dumps(self.facts, ensure_ascii=False, indent=2)}\n\n"
            f"Последние сообщения:\n{self.format_messages(self.get_active_history()[-RECENT_MESSAGES_LIMIT:])}\n\n"
            f"Новое сообщение пользователя:\n{message}\n\n"
            "Верни обновленные facts как JSON object со строковыми значениями."
        )
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            raw_facts = response.json()["choices"][0]["message"]["content"].strip()
            if raw_facts.startswith("```json"):
                raw_facts = raw_facts[7:]
            if raw_facts.startswith("```"):
                raw_facts = raw_facts[3:]
            if raw_facts.endswith("```"):
                raw_facts = raw_facts[:-3]
            parsed_facts = json.loads(raw_facts.strip())
            validate(instance=parsed_facts, schema=schema)
            if not isinstance(parsed_facts, dict):
                raise ValueError("facts должен быть JSON object")
        except Exception as e:
            print(f"Не удалось обновить facts, продолжаем со старыми facts: {e}")
            return

        self.facts = {
            str(key): (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
            for key, value in parsed_facts.items()
        }
        self.save_facts()

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    def print_last_token_counts(self) -> None:
        if self.last_token_counts is not None:
            self.print_token_counts(self.last_token_counts)
            self.last_token_counts = None

    def check_server(self) -> bool:
        """Проверяет, доступен ли сервер"""
        try:
            requests.options(self.api_url, timeout=2)
            return True
        except requests.RequestException:
            try:
                parsed_url = urlparse(self.api_base_url)
                host = parsed_url.hostname
                port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
                if not host:
                    return False
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            except OSError:
                return False

    def server_hint(self) -> str:
        return (
            f"Сейчас клиент настроен на {self.api_url}\n"
            "Для сервера на другой машине запустите клиент так:\n"
            "   python3 main.py --server 192.168.1.50:8080\n"
            "или через переменную окружения:\n"
            "   AI_ADVENT_API_BASE=http://192.168.1.50:8080 python3 main.py"
        )

    def get_temperature_from_user(self) -> float:
        """Запрашивает температуру у пользователя. Enter -> значение по умолчанию"""
        user_input = input(f"Температура (Enter для {DEFAULT_TEMPERATURE}): ").strip()
        if user_input == "":
            return DEFAULT_TEMPERATURE
        try:
            temp = float(user_input)
            return max(0.0, min(2.0, temp))
        except ValueError:
            print(f"Некорректный ввод, используем {DEFAULT_TEMPERATURE}")
            return DEFAULT_TEMPERATURE

    def send_with_progress(self, payload: Dict[str, Any]) -> requests.Response:
        """Отправляет запрос с индикатором прогресса"""
        print("Модель думает", end="", flush=True)

        stop_animation: bool = False

        def show_progress() -> None:
            chars: List[str] = [".", "..", "...", "...."]
            i: int = 0
            while not stop_animation:
                print(f"\rМодель думает{chars[i % len(chars)]}", end="", flush=True)
                i += 1
                time.sleep(0.3)

        progress_thread: Thread = Thread(target=show_progress)
        progress_thread.start()

        try:
            response: requests.Response = requests.post(
                self.api_url, json=payload, timeout=60
            )
            stop_animation = True
            progress_thread.join(timeout=0.5)
            print("\r" + " " * 30 + "\r", end="", flush=True)
            return response
        except Exception as e:
            stop_animation = True
            progress_thread.join(timeout=0.5)
            print("\r" + " " * 30 + "\r", end="", flush=True)
            raise e

    def get_json_response(
        self, message: str, temperature: float, schema: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Получает ответ в JSON формате с валидацией по схеме."""
        schema_to_use = schema or self.RESPONSE_SCHEMA
        raw_response: str = ""

        refusal = self.build_invariant_refusal(message)
        if refusal:
            parsed_json: Dict[str, Any] = {
                "answer": refusal,
                "confidence": 1.0,
                "intent": "command",
                "data": {"blocked_by_invariant": True},
            }
            validate(instance=parsed_json, schema=schema_to_use)
            serialized = json.dumps(parsed_json, ensure_ascii=False)
            self.record_local_response(message, serialized)
            self.last_token_counts = self.build_token_counts(message, serialized)
            print(
                f"\nJSON ответ: {json.dumps(parsed_json, indent=2, ensure_ascii=False)}"
            )
            return parsed_json

        system_prompt = f"""
Ты - ассистент, который возвращает ответы строго в JSON формате.

Твоя задача - преобразовать ответ в JSON объект, соответствующий этой схеме:
{json.dumps(schema_to_use, indent=2, ensure_ascii=False)}

Правила:
1. Возвращай ТОЛЬКО JSON объект, без любого другого текста
2. Не используй markdown или другие обёртки
3. Все строки должны быть в двойных кавычках
4. Убедись, что JSON валидный
5. Поле 'answer' содержит основной ответ на вопрос
6. Поле 'confidence' показывает уверенность в ответе (0-1)
7. Поле 'intent' определяет тип запроса (question/command/statement)
8. Поле 'data' может содержать дополнительные данные
"""
        system_prompt = self.build_stateful_system_prompt(system_prompt)
        rag_context = self.search_rag(message)
        if rag_context and rag_context.get("weak_context") and self.rag_strict_unknown:
            parsed_json = {
                "answer": "Не знаю: в локальной базе не найден достаточно релевантный контекст. Пожалуйста, уточните вопрос.",
                "confidence": 0.0,
                "intent": "question",
                "data": {
                    "rag_weak_context": True,
                    "rag_best_score": rag_context.get("best_score", 0.0),
                    "rag_threshold": self.rag_score_threshold,
                    "sources": [],
                    "quotes": [],
                },
            }
            validate(instance=parsed_json, schema=schema_to_use)
            serialized = json.dumps(parsed_json, ensure_ascii=False)
            self.record_local_response(message, serialized)
            self.last_token_counts = self.build_token_counts(message, serialized)
            print(
                f"\nJSON ответ: {json.dumps(parsed_json, indent=2, ensure_ascii=False)}"
            )
            return parsed_json
        system_prompt = self.build_rag_system_prompt(system_prompt, rag_context)

        user_message = (
            f"Вопрос: {message}\n\nВерни ответ строго в указанном JSON формате."
        )

        self.append_message("user", message)
        self.trim_active_history_if_needed()
        self.update_facts_from_user_message(message)

        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}]
            + self.build_context_messages()[:-1]
            + [{"role": "user", "content": user_message}],
            "max_tokens": 50000,
            "temperature": temperature,
        }

        start_time = time.time()

        try:
            response: requests.Response = self.send_with_progress(payload)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()

            elapsed_time = time.time() - start_time

            raw_response: str = result["choices"][0]["message"]["content"].strip()

            # Очищаем ответ от возможной разметки
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]

            raw_response = raw_response.strip()

            # Парсим JSON
            parsed_json: Dict[str, Any] = json.loads(raw_response)

            # Валидируем по схеме
            validate(instance=parsed_json, schema=schema_to_use)

            violations = self.check_invariants(
                json.dumps(parsed_json, ensure_ascii=False)
            )
            if violations:
                correction = (
                    "Предыдущий JSON нарушил обязательные инварианты:\n- "
                    + "\n- ".join(violations)
                    + "\nВерни исправленный JSON по исходной схеме, без markdown."
                )
                retry_payload = dict(payload)
                retry_payload["messages"] = payload["messages"] + [
                    {"role": "assistant", "content": raw_response},
                    {"role": "user", "content": correction},
                ]
                retry_response = self.send_with_progress(retry_payload)
                retry_response.raise_for_status()
                raw_response = retry_response.json()["choices"][0]["message"]["content"].strip()
                raw_response = raw_response.removeprefix("```json").removeprefix("```")
                raw_response = raw_response.removesuffix("```").strip()
                parsed_json = json.loads(raw_response)
                validate(instance=parsed_json, schema=schema_to_use)
                remaining = self.check_invariants(raw_response)
                if remaining:
                    print(
                        "\nОтвет заблокирован: модель повторно нарушила инварианты: "
                        + "; ".join(remaining)
                    )
                    self.save_conversation_state()
                    return None

            if rag_context:
                parsed_json.setdefault("data", {})
                if isinstance(parsed_json["data"], dict):
                    parsed_json["data"]["rag_best_score"] = rag_context["best_score"]
                    parsed_json["data"]["rag_sources"] = [
                        {
                            "source_path": hit["source_path"],
                            "chunk_index": hit["chunk_index"],
                            "section": hit.get("heading_path") or [],
                            "score": hit["score"],
                            "rerank_score": hit.get("rerank_score"),
                        }
                        for hit in rag_context["hits"]
                    ]
                    parsed_json["data"]["rag_quotes"] = [
                        {
                            "source_path": hit["source_path"],
                            "chunk_index": hit["chunk_index"],
                            "quote": self.quote_from_hit(hit),
                        }
                        for hit in rag_context["hits"]
                    ]

            print(
                f"\nJSON ответ: {json.dumps(parsed_json, indent=2, ensure_ascii=False)}"
            )
            print(f"⏱️ Время ответа: {elapsed_time:.2f} сек")

            self.append_message(
                "assistant",
                json.dumps(parsed_json, ensure_ascii=False),
            )
            self.trim_active_history_if_needed()
            self.last_token_counts = self.build_token_counts(message, raw_response)
            self.save_conversation_state()

            return parsed_json

        except json.JSONDecodeError as e:
            print(f"\nОшибка парсинга JSON: {e}")
            print(f"Ответ модели: {raw_response}")
            return None
        except ValidationError as e:
            print(f"\nОшибка валидации схемы: {e}")
            return None
        except requests.exceptions.Timeout:
            print("\nОшибка: Превышено время ожидания ответа от модели")
            return None
        except requests.exceptions.ConnectionError:
            print("\nОшибка: Не удалось подключиться к серверу")
            return None
        except Exception as e:
            print(f"\nОшибка: {e}")
            return None

    def send_message_streaming(
        self, message: str, temperature: float, system_prompt: str = "Ты полезный ассистент."
    ) -> Optional[str]:
        """Отправляет сообщение с потоковой передачей ответа, показывает время и токены."""

        if self.invariants:
            print(
                "Строгие инварианты активны: используется буферизованный режим, "
                "чтобы проверить ответ до показа."
            )
            return self.send_message_simple(message, temperature, system_prompt)

        rag_context = self.search_rag(message)
        if rag_context and rag_context.get("weak_context") and self.rag_strict_unknown:
            print(
                "RAG: контекст ниже порога "
                f"(score {rag_context.get('best_score', 0.0):.3f}), "
                "ответ будет без streaming."
            )
            return self.send_message_simple(
                message, temperature, system_prompt, rag_context=rag_context
            )
        if rag_context:
            print(
                "RAG: найден релевантный контекст "
                f"(score {rag_context['best_score']:.3f}, "
                f"{rag_context.get('filtered_hit_count', len(rag_context['hits']))}/"
                f"{rag_context.get('raw_hit_count', len(rag_context['hits']))} chunks), "
                "ответ будет без streaming."
            )
            return self.send_message_simple(
                message, temperature, system_prompt, rag_context=rag_context
            )

        self.append_message("user", message)
        self.trim_active_history_if_needed()
        self.update_facts_from_user_message(message)

        stateful_prompt = self.build_stateful_system_prompt(system_prompt)
        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": stateful_prompt}]
            + self.build_context_messages(),
            "max_tokens": 50000,
            "temperature": temperature,
            "stream": True,
        }

        start_time = time.time()
        print("\nАссистент: ", end="", flush=True)

        full_response: str = ""

        try:
            response: requests.Response = requests.post(
                self.api_url, json=payload, stream=True, timeout=60
            )

            for line in response.iter_lines():
                if line:
                    line_str: str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data: str = line_str[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk: Dict[str, Any] = json.loads(data)
                            if "choices" in chunk and chunk["choices"][0].get("delta", {}).get("content"):
                                content: str = chunk["choices"][0]["delta"]["content"]
                                print(content, end="", flush=True)
                                full_response += content
                        except:
                            pass

            elapsed_time = time.time() - start_time
            print()

            violations = self.check_invariants(full_response)
            if violations:
                print("Предупреждение: ответ нарушает инварианты: " + "; ".join(violations))

            self.append_message("assistant", full_response)
            self.trim_active_history_if_needed()
            token_counts = self.build_token_counts(message, full_response)
            self.save_conversation_state()

            print(f"⏱️ Время ответа: {elapsed_time:.2f} сек")
            self.print_token_counts(token_counts)
            return full_response

        except Exception as e:
            print(f"\nОшибка: {e}")
            return None

    def send_message_simple(
        self,
        message: str,
        temperature: float,
        system_prompt: str = "Ты полезный ассистент.",
        rag_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Отправляет сообщение без потокового вывода, показывает время и токены"""
        refusal = self.build_invariant_refusal(message)
        if refusal:
            self.record_local_response(message, refusal)
            print(f"\nАссистент: {refusal}")
            token_counts = self.build_token_counts(message, refusal)
            self.print_token_counts(token_counts)
            return refusal

        if rag_context is None:
            rag_context = self.search_rag(message)
        if rag_context and rag_context.get("weak_context") and self.rag_strict_unknown:
            assistant_message = self.build_weak_context_response(rag_context)
            self.record_local_response(message, assistant_message)
            print(f"\nАссистент: {assistant_message}")
            token_counts = self.build_token_counts(message, assistant_message)
            self.print_token_counts(token_counts)
            return assistant_message
        if rag_context:
            print(
                f"RAG: используется локальная база, score {rag_context['best_score']:.3f}, "
                f"chunks {rag_context.get('filtered_hit_count', len(rag_context['hits']))}/"
                f"{rag_context.get('raw_hit_count', len(rag_context['hits']))}"
            )

        self.append_message("user", message)
        self.trim_active_history_if_needed()
        self.update_facts_from_user_message(message)

        stateful_prompt = self.build_stateful_system_prompt(system_prompt)
        stateful_prompt = self.build_rag_system_prompt(stateful_prompt, rag_context)
        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": stateful_prompt}]
            + self.build_context_messages(),
            "max_tokens": 50000,
            "temperature": temperature,
        }

        start_time = time.time()

        try:
            response: requests.Response = self.send_with_progress(payload)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()

            elapsed_time = time.time() - start_time

            assistant_message: str = result["choices"][0]["message"]["content"]
            violations = self.check_invariants(assistant_message)
            if violations:
                correction = (
                    "Исправь ответ: он нарушает обязательные инварианты:\n- "
                    + "\n- ".join(violations)
                    + "\nВерни только исправленный ответ."
                )
                retry_payload = dict(payload)
                retry_payload["messages"] = payload["messages"] + [
                    {"role": "assistant", "content": assistant_message},
                    {"role": "user", "content": correction},
                ]
                retry_response = self.send_with_progress(retry_payload)
                retry_response.raise_for_status()
                assistant_message = retry_response.json()["choices"][0]["message"]["content"]
                remaining = self.check_invariants(assistant_message)
                if remaining:
                    print(
                        "\nОтвет заблокирован: модель повторно нарушила инварианты: "
                        + "; ".join(remaining)
                    )
                    self.save_conversation_state()
                    return None
            if rag_context and (
                "Источники:" not in assistant_message
                or "Цитаты:" not in assistant_message
            ):
                assistant_message = self.format_rag_required_sections(
                    assistant_message, rag_context
                )
            print(f"\nАссистент: {assistant_message}")

            self.append_message("assistant", assistant_message)
            self.trim_active_history_if_needed()
            token_counts = self.build_token_counts(message, assistant_message)
            self.save_conversation_state()

            print(f"⏱️ Время ответа: {elapsed_time:.2f} сек")
            self.print_token_counts(token_counts)
            return assistant_message

        except requests.exceptions.Timeout:
            print("\nОшибка: Превышено время ожидания ответа от модели")
            return None
        except requests.exceptions.ConnectionError:
            print("\nОшибка: Не удалось подключиться к серверу. Запустите сервер командой:")
            print("   ./start_server.sh medium")
            print(self.server_hint())
            return None
        except Exception as e:
            print(f"\nОшибка: {e}")
            return None

    def clear_history(self) -> None:
        """Очищает только краткосрочную память активного пользователя."""
        self.conversation_history = []
        self.conversation_summary = ""
        self.branches = {}
        self.current_branch = None
        self.save_memories()
        print("Краткосрочная память текущего диалога очищена")

    def create_checkpoint(self) -> None:
        if self.context_strategy != CONTEXT_STRATEGY_BRANCHES:
            print("Checkpoint доступен только в стратегии 3: ветки от checkpoint")
            return

        checkpoint_history = list(self.get_active_history())
        self.branches = {
            "branch_1": [dict(message) for message in checkpoint_history],
            "branch_2": [dict(message) for message in checkpoint_history],
        }
        self.current_branch = "branch_1"
        self.save_branches()
        print("Checkpoint создан. Доступны branch_1 и branch_2, активна branch_1")

    def switch_branch(self, branch_name: str) -> None:
        if self.context_strategy != CONTEXT_STRATEGY_BRANCHES:
            print("Переключение веток доступно только в стратегии 3")
            return
        if branch_name not in self.branches:
            print(f"Ветка '{branch_name}' не найдена. Используйте /branches")
            return
        self.current_branch = branch_name
        self.save_branches()
        print(f"Активная ветка: {branch_name}")

    def show_branches(self) -> None:
        if self.context_strategy != CONTEXT_STRATEGY_BRANCHES:
            print("Ветки доступны только в стратегии 3")
            return
        if not self.branches:
            print("Веток пока нет. Создайте checkpoint командой /checkpoint")
            return
        print("\nВетки:")
        for name, history in self.branches.items():
            marker = "*" if name == self.current_branch else " "
            print(f" {marker} {name}: сообщений {len(history)}")
        print()

    def set_profile_value(self, expression: str) -> None:
        if "=" not in expression:
            print("Формат: /profile <поле>=<значение>")
            return
        key, value = (part.strip() for part in expression.split("=", 1))
        if not key or not value:
            print("Поле и значение не должны быть пустыми")
            return
        self.profile[key] = value
        self.save_profile()
        print(f"Профиль {self.current_user!r} обновлён: {key}")

    def switch_user(self, user_id: str) -> None:
        user_id = user_id.strip()
        if not user_id:
            print("Формат: /user <user_id>")
            return
        if len(user_id) > 100:
            print("user_id не должен быть длиннее 100 символов")
            return
        created = user_id not in self.profiles
        self.activate_user_memory(user_id)
        self.normalize_invariants()
        self.current_user = user_id
        self.profile = self.profiles.setdefault(user_id, {})
        self.trim_active_history_if_needed()
        self.save_conversation_state()
        action = "создан и выбран" if created else "выбран"
        print(f"Пользователь {user_id!r} {action}")

    def show_users(self) -> None:
        print("\nПОЛЬЗОВАТЕЛИ:")
        for user_id in self.profiles:
            marker = "*" if user_id == self.current_user else " "
            print(f" {marker} {user_id}")
        print()

    def setup_profile(self) -> None:
        """Короткое интервью для первоначальной персонализации."""
        print(
            f"\nНастройка профиля {self.current_user!r}. "
            "Пустой ответ оставляет поле без изменений."
        )
        questions = {
            "style": "Предпочтительный стиль ответов: ",
            "constraints": "Постоянные ограничения и технологии: ",
            "context": "Ваш контекст и цель работы с агентом: ",
        }
        for key, question in questions.items():
            value = input(question).strip()
            if value:
                self.profile[key] = value
        self.save_profile()
        print("Профиль сохранён\n")

    def show_profile(self) -> None:
        print(f"\nПРОФИЛЬ {self.current_user!r}:")
        print(json.dumps(self.profile, ensure_ascii=False, indent=2) if self.profile else "пуст")
        print()

    def create_task(self, description: str) -> None:
        description = description.strip()
        if not description:
            print("Формат: /task <описание>")
            return
        self.task_state = {
            "description": description,
            "stage": "collecting",
            "plan": "",
        }
        self.save_task_state()
        print("Задача создана. Стадия: collecting")

    def show_task(self) -> None:
        if not self.task_state.get("stage"):
            print("\nАктивной задачи нет\n")
            return
        print("\nЗАДАЧА:")
        print(json.dumps(self.task_state, ensure_ascii=False, indent=2))
        print()

    def set_task_plan(self, plan: str) -> None:
        if not self.task_state.get("stage"):
            print("Сначала создайте задачу: /task <описание>")
            return
        if not plan.strip():
            print("Формат: /plan <текст плана>")
            return
        self.task_state["plan"] = plan.strip()
        self.save_task_state()
        print("План задачи сохранён")

    def set_memory_value(self, layer: str, expression: str) -> None:
        if "=" not in expression:
            print(f"Формат: /{layer} <ключ>=<значение>")
            return
        key, value = (part.strip() for part in expression.split("=", 1))
        if not key or not value:
            print("Ключ и значение не должны быть пустыми")
            return
        targets = {
            "work": self.working_notes,
            "remember": self.facts,
            "decision": self.decisions,
        }
        targets[layer][key] = value
        self.save_memories()
        print(f"Значение сохранено в слой {layer}: {key}")

    def clear_working_memory(self) -> None:
        self.task_state = {"description": "", "stage": None, "plan": ""}
        self.working_notes = {}
        self.save_memories()
        print("Рабочая память текущего пользователя очищена")

    def forget_memory_value(self, layer: str, key: str) -> None:
        key = key.strip()
        targets = {
            "work": [self.working_notes],
            "long": [self.facts, self.decisions],
        }
        removed = False
        for target in targets[layer]:
            if key in target:
                del target[key]
                removed = True
        if not removed:
            print(f"Ключ {key!r} в слое {layer} не найден")
            return
        self.save_memories()
        print(f"Ключ {key!r} удалён из слоя {layer}")

    def show_memory(self) -> None:
        active_history = self.get_active_history()
        view = {
            "user": self.current_user,
            "short_term": {
                "messages": len(active_history),
                "recent": active_history[-RECENT_MESSAGES_LIMIT:],
                "current_branch": self.current_branch,
            },
            "working": {
                "task": self.task_state,
                "notes": self.working_notes,
            },
            "long_term": {
                "profile": self.profile,
                "decisions": self.decisions,
                "knowledge": self.facts,
                "invariants": self.invariants,
            },
        }
        print("\nПАМЯТЬ АКТИВНОГО ПОЛЬЗОВАТЕЛЯ:")
        print(json.dumps(view, ensure_ascii=False, indent=2))
        print()

    def add_invariant(self, expression: str) -> None:
        parts = [part.strip() for part in expression.split("|", 1)]
        rule = parts[0]
        if not rule:
            print("Формат: /invariant <правило> | <запрещённые термины через запятую>")
            return
        forbidden = []
        if len(parts) == 2:
            forbidden = [term.strip() for term in parts[1].split(",") if term.strip()]
        if not forbidden:
            forbidden = self.infer_forbidden_terms(rule)
        if not forbidden:
            print(
                "Инвариант не добавлен: нельзя построить строгую проверку. "
                "Укажите запрещённые термины после |"
            )
            return
        self.invariants.append({"rule": rule, "forbidden_terms": forbidden})
        self.save_invariants()
        print("Инвариант добавлен")

    def show_invariants(self) -> None:
        print("\nИНВАРИАНТЫ:")
        print(json.dumps(self.invariants, ensure_ascii=False, indent=2) if self.invariants else "пусто")
        print()

    def show_help(self) -> None:
        print("\n" + "=" * 50)
        print("Команды:")
        print("  /clear   - Очистить только краткосрочную память")
        print("  /history - Показать историю диалога")
        print("  /memory  - Показать все три слоя памяти")
        print("  /status  - Проверить статус сервера")
        print("  /rag-status - Проверить настройки RAG")
        print("  /json    - Режим JSON ответов")
        print("  /normal  - Обычный режим ответов")
        print("  /facts   - Показать facts (стратегия 2)")
        print("  /profile             - Показать профиль")
        print("  /profile поле=значение - Изменить профиль")
        print("  /setup               - Пройти интервью персонализации")
        print("  /users               - Показать пользователей")
        print("  /user <user_id>      - Создать или выбрать пользователя")
        print("  /task                 - Показать активную задачу")
        print("  /task <описание>      - Создать новую задачу")
        print("  /plan <текст>         - Сохранить план задачи")
        print("  /work ключ=значение   - Записать данные текущей задачи")
        print("  /work-forget ключ     - Удалить рабочую заметку")
        print("  /work-clear           - Очистить задачу и рабочие заметки")
        print("  /remember ключ=значение - Сохранить долгосрочное знание")
        print("  /decision ключ=значение - Сохранить долгосрочное решение")
        print("  /forget ключ          - Удалить знание или решение")
        print(f"  /stage <стадия>       - Перейти по state machine ({', '.join(TASK_STAGES)})")
        print("  /invariants           - Показать инварианты")
        print("  /invariant правило | запрещённые термины - Добавить инвариант")
        print("  /checkpoint       - Создать две ветки от текущего диалога (стратегия 3)")
        print("  /branches         - Показать ветки (стратегия 3)")
        print("  /branch <name>    - Переключиться на ветку (стратегия 3)")
        print("  /mcp-start        - Запустить MCP-сервер мировых новостей")
        print("  /mcp-tools        - Показать инструменты MCP-сервера")
        print("  /mcp-call <tool> [JSON] - Вызвать инструмент MCP-сервера")
        print("  /mcp-stop         - Остановить MCP-сервер")
        print("  /summary-now      - Сразу обновить periodic summary")
        print("  /summary-status   - Статус periodic summary")
        print("  /code-review <папка> - MCP review исходного кода в папке")
        print("  /code-review-deep <папка> - Более глубокое review большего числа файлов")
        print("  /bugs-write       - Записать проблемы в <review-папка>/bugs/*.txt")
        print("  /bugs-fix [папка] - Исправить баги из <папка>/bugs/*.txt")
        print("  Также можно: сделай ревью /path/to/project")
        print("  Также можно: запиши баги в папку")
        print("  Также можно: исправь баги")
        print("  /help    - Показать эту справку")
        print("  /exit    - Выйти из программы")
        print("=" * 50 + "\n")

    def show_history(self) -> None:
        active_history = self.get_active_history()
        if not active_history:
            print("\nИстория пуста\n")
            return
        print("\n" + "=" * 50)
        print("ИСТОРИЯ ДИАЛОГА:")
        print("=" * 50)
        print(f"Стратегия контекста: {self.context_strategy}")
        print(f"Активный пользователь: {self.current_user}")
        if self.current_branch:
            print(f"Активная ветка: {self.current_branch}")
        print(f"ПОСЛЕДНИЕ СООБЩЕНИЯ (до {RECENT_MESSAGES_LIMIT}):")
        for i, msg in enumerate(active_history[-RECENT_MESSAGES_LIMIT:], 1):
            role_label = "Пользователь" if msg["role"] == "user" else "Ассистент"
            content_preview = msg["content"][:100]
            if len(msg["content"]) > 100:
                content_preview += "..."
            print(f"{role_label}: {content_preview}")
        print("=" * 50 + "\n")

    def check_status(self) -> None:
        print("\nПроверка статуса...")
        if self.check_server():
            print(f"Сервер доступен: {self.api_base_url}")
            try:
                test_payload = {
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                }
                response = requests.post(self.api_url, json=test_payload, timeout=5)
                if response.status_code == 200:
                    print("Модель отвечает на запросы")
                else:
                    print(f"Сервер ответил с кодом: {response.status_code}")
            except:
                print("Модель не отвечает на тестовый запрос")
        else:
            print("Сервер недоступен! Запустите сервер командой:")
            print("   ./start_server.sh medium")
            print(self.server_hint())
        print()

    def show_rag_status(self) -> None:
        qdrant_url = os.environ.get("AI_ADVENT_RAG_QDRANT_URL") or self.get_rag_setting(
            "qdrant", "url", "http://localhost:6333"
        )
        collection = os.environ.get("AI_ADVENT_RAG_COLLECTION") or self.get_rag_setting(
            "qdrant", "collection_name", "local_knowledge_base"
        )
        ollama_url = os.environ.get("AI_ADVENT_RAG_OLLAMA_URL") or self.get_rag_setting(
            "embedding", "ollama_url", "http://localhost:11434"
        )
        model = os.environ.get("AI_ADVENT_RAG_EMBEDDING_MODEL") or self.get_rag_setting(
            "embedding", "embedding_model", "nomic-embed-text"
        )
        try:
            from qdrant_client import QdrantClient

            qdrant_installed = True
            try:
                client = QdrantClient(url=qdrant_url)
                collections = {item.name for item in client.get_collections().collections}
                qdrant_ready = collection in collections
            except Exception:
                qdrant_ready = False
        except ImportError:
            qdrant_installed = False
            qdrant_ready = False

        print("\nRAG:")
        print(f"Включен: {'да' if self.rag_enabled else 'нет'}")
        print(f"Qdrant URL: {qdrant_url}")
        print(f"Collection: {collection}")
        print(f"Ollama embeddings: {ollama_url} / {model}")
        print(f"Threshold: {self.rag_score_threshold}")
        print(f"Top-K до фильтрации: {self.rag_pre_top_k}")
        print(f"Top-K после фильтрации: {self.rag_top_k}")
        print(f"Query rewrite: {'да' if self.rag_query_rewrite_enabled else 'нет'}")
        print(f"Строгое 'не знаю' ниже порога: {'да' if self.rag_strict_unknown else 'нет'}")
        print(f"Максимальная длина цитаты: {self.rag_quote_chars} символов")
        print(f"qdrant-client установлен: {'да' if qdrant_installed else 'нет'}")
        print(f"Коллекция доступна: {'да' if qdrant_ready else 'нет'}")
        print()

    def show_facts(self) -> None:
        if not self.facts:
            print("\nДолговременные знания пусты\n")
            return
        print("\nДОЛГОВРЕМЕННЫЕ ЗНАНИЯ:")
        print(json.dumps(self.facts, ensure_ascii=False, indent=2))
        print()

    def start_mcp_server(self) -> None:
        try:
            result = self.mcp_client.start()
            if result.get("already_running"):
                print("\nMCP-сервер уже запущен\n")
                return
            server = result.get("serverInfo", {})
            print(
                f"\nMCP-сервер запущен: {server.get('name', 'unknown')} "
                f"{server.get('version', '')}\n"
            )
        except (MCPClientError, OSError) as error:
            print(f"\nНе удалось запустить MCP-сервер: {error}\n")

    def show_mcp_tools(self) -> None:
        try:
            tools = self.mcp_client.list_tools()
            print("\nИНСТРУМЕНТЫ MCP-СЕРВЕРА:")
            for tool in tools:
                print(f"- {tool.get('name')}: {tool.get('description', '')}")
                print(
                    "  Аргументы: "
                    + json.dumps(tool.get("inputSchema", {}), ensure_ascii=False)
                )
            if not tools:
                print("нет инструментов")
            print()
        except MCPClientError as error:
            print(f"\n{error}\n")

    def call_mcp_tool(self, expression: str) -> None:
        parts = expression.strip().split(maxsplit=1)
        if not parts:
            print("Формат: /mcp-call <tool> [JSON-аргументы]")
            return
        name = parts[0]
        try:
            arguments = json.loads(parts[1]) if len(parts) == 2 else {}
            if not isinstance(arguments, dict):
                raise ValueError("аргументы должны быть JSON-объектом")
            result = self.mcp_client.call_tool(name, arguments)
            for content in result.get("content", []):
                if content.get("type") == "text":
                    print(f"\n{content.get('text', '')}\n")
            if result.get("isError"):
                print("Инструмент завершился с ошибкой\n")
        except (json.JSONDecodeError, ValueError) as error:
            print(f"\nНекорректные аргументы: {error}\n")
        except MCPClientError as error:
            print(f"\nОшибка MCP: {error}\n")

    def start_code_review_mcp(self) -> None:
        if self.code_review_mcp_client.is_running:
            return
        self.code_review_mcp_client.start()

    def default_review_folder(self) -> str:
        return str(Path(__file__).resolve().parent)

    def extract_review_folder_from_text(self, text: str) -> str:
        return self.extract_optional_folder_from_text(text) or self.default_review_folder()

    def extract_optional_folder_from_text(self, text: str) -> str:
        stripped = text.strip()
        quoted = re.search(r"[\"'«“](.*?)[\"'»”]", stripped)
        if quoted and quoted.group(1).strip():
            return quoted.group(1).strip()

        tokens = stripped.split()
        for token in reversed(tokens):
            cleaned = token.strip(".,;:()[]{}")
            if (
                "/" in cleaned
                or cleaned.startswith("~")
                or cleaned.startswith(".")
            ):
                return cleaned
        return ""

    def is_natural_review_request(self, text: str) -> bool:
        normalized = text.casefold()
        review_terms = (
            "сделай ревью",
            "сделать ревью",
            "проведи ревью",
            "провести ревью",
            "ревью кода",
            "код ревью",
            "code review",
            "review code",
        )
        flexible_review_patterns = (
            r"\bсдела(?:й|ть)\b.*\bревью\b",
            r"\bпровед(?:и|ите|ение|ести)\b.*\bревью\b",
            r"\breview\b.*\bcode\b",
            r"\bcode\b.*\breview\b",
        )
        return any(term in normalized for term in review_terms) or any(
            re.search(pattern, normalized) for pattern in flexible_review_patterns
        )

    def is_deep_review_request(self, text: str) -> bool:
        normalized = text.casefold()
        return self.is_natural_review_request(text) and any(
            term in normalized for term in ("глубок", "подробн", "полное", "deep")
        )

    def is_natural_bug_write_request(self, text: str) -> bool:
        normalized = text.casefold()
        has_bug_term = any(term in normalized for term in ("баг", "bug", "проблем"))
        has_write_term = any(
            term in normalized
            for term in (
                "запиши",
                "записать",
                "создай",
                "создать",
                "сохрани",
                "сохранить",
                "выведи",
                "вывести",
                "добавь",
                "добавить",
            )
        )
        has_folder_term = any(term in normalized for term in ("папк", "файл", "bugs"))
        return has_bug_term and has_write_term and has_folder_term

    def is_natural_bug_fix_request(self, text: str) -> bool:
        normalized = text.casefold()
        has_bug_term = any(term in normalized for term in ("баг", "bug", "проблем"))
        has_fix_term = any(
            term in normalized
            for term in (
                "исправь",
                "исправить",
                "почини",
                "починить",
                "зафикси",
                "фикс",
                "fix",
            )
        )
        return has_bug_term and has_fix_term

    def review_code_folder(self, folder_expression: str, deep: bool = False) -> None:
        folder = folder_expression.strip() or self.default_review_folder()
        if not folder:
            print("Формат: /code-review <папка>")
            return
        try:
            arguments: Dict[str, Any] = {
                "api_url": self.api_url,
                "folder_path": folder,
            }
            if deep:
                arguments.update(
                    {
                        "max_files": 300,
                        "max_chars_per_file": 10000,
                        "batch_chars": 30000,
                    }
                )
            self.start_code_review_mcp()
            result = self.code_review_mcp_client.call_tool(
                "review_code_folder",
                arguments,
            )
            for content in result.get("content", []):
                if content.get("type") == "text":
                    print(f"\n{content.get('text', '')}\n")
            structured = result.get("structuredContent", {})
            problems = structured.get("problems", [])
            self.last_code_review_problems = (
                problems if isinstance(problems, list) else []
            )
            reviewed_folder = structured.get("folder")
            self.last_code_review_folder = (
                reviewed_folder if isinstance(reviewed_folder, str) else folder
            )
            if self.last_code_review_problems:
                print(
                    "Чтобы записать проблемы в файлы, выполните: /bugs-write"
                )
        except (MCPClientError, OSError) as error:
            print(f"\nОшибка code review MCP: {error}\n")

    def write_bug_reports(self) -> None:
        if not self.last_code_review_problems:
            print("\nНет сохраненного списка проблем. Сначала выполните /code-review <папка>\n")
            return
        if not self.last_code_review_folder:
            print("\nНеизвестна папка последнего review. Сначала выполните /code-review <папка>\n")
            return
        try:
            self.start_code_review_mcp()
            result = self.code_review_mcp_client.call_tool(
                "write_bug_reports",
                {
                    "project_dir": self.last_code_review_folder,
                    "problems": self.last_code_review_problems,
                },
            )
            for content in result.get("content", []):
                if content.get("type") == "text":
                    print(f"\n{content.get('text', '')}\n")
        except (MCPClientError, OSError) as error:
            print(f"\nОшибка записи bug-файлов: {error}\n")

    def fix_bugs_from_folder(self, folder_expression: str = "") -> None:
        folder = folder_expression.strip()
        if not folder:
            folder = self.last_code_review_folder or self.default_review_folder()
        try:
            self.start_code_review_mcp()
            result = self.code_review_mcp_client.call_tool(
                "fix_bugs_from_folder",
                {
                    "api_url": self.api_url,
                    "project_dir": folder,
                },
            )
            for content in result.get("content", []):
                if content.get("type") == "text":
                    print(f"\n{content.get('text', '')}\n")
        except (MCPClientError, OSError) as error:
            print(f"\nОшибка исправления bug-файлов: {error}\n")

    def run(self, use_streaming: bool = False, use_json_mode: bool = False) -> None:
        print("\n" + "=" * 50)
        print("ДОБРО ПОЖАЛОВАТЬ В ЧАТ С LOCAL LLM")
        print("=" * 50)
        print(f"Модель: Qwen3-4B-Instruct (локальная)")
        print(f"Режим: {'потоковый' if use_streaming else 'обычный'}")
        print(f"Формат ответа: {'JSON' if use_json_mode else 'текстовый'}")
        print(f"Стратегия контекста: {self.context_strategy}")
        print(f"Активный пользователь: {self.current_user}")
        print(f"LLM API: {self.api_url}")
        print(
            "RAG: "
            f"{'включен' if self.rag_enabled else 'выключен'}, "
            f"threshold {self.rag_score_threshold}, "
            f"top_k {self.rag_pre_top_k}->{self.rag_top_k}, "
            f"rewrite {'on' if self.rag_query_rewrite_enabled else 'off'}"
        )
        print("Введите /help для списка команд")
        print(f"Файл памяти: {self.memory_file}")
        active_history = self.get_active_history()
        if active_history:
            print(f"Загружено сообщений из истории: {len(active_history)}")
        if self.current_branch:
            print(f"Активная ветка: {self.current_branch}")
        if self.task_state.get("stage"):
            print(
                f"Возобновлена задача [{self.task_state['stage']}]: "
                f"{self.task_state['description']}"
            )
        print("=" * 50 + "\n")

        if not self.check_server():
            print("СЕРВЕР НЕ ДОСТУПЕН!")
            print("\nЗапустите сервер в другом терминале:")
            print("cd ~/code/AI_advent")
            print("./start_server.sh medium")
            print()
            print(self.server_hint())
            print("\nПосле запуска сервера, перезапустите эту программу.")
            return

        print("Сервер доступен! Готов к работе.\n")
        self.start_periodic_summary()

        while self.running:
            try:
                user_input = input("Вы: ").strip()

                if user_input.lower() in ["/exit", "/quit", "выход", "exit", "quit"]:
                    print("\nДо свидания!")
                    break
                elif user_input == "/clear":
                    self.clear_history()
                    continue
                elif user_input == "/history":
                    self.show_history()
                    continue
                elif user_input == "/memory":
                    self.show_memory()
                    continue
                elif user_input == "/help":
                    self.show_help()
                    continue
                elif user_input == "/status":
                    self.check_status()
                    continue
                elif user_input == "/rag-status":
                    self.show_rag_status()
                    continue
                elif user_input == "/facts":
                    self.show_facts()
                    continue
                elif user_input == "/profile":
                    self.show_profile()
                    continue
                elif user_input == "/users":
                    self.show_users()
                    continue
                elif user_input.startswith("/user "):
                    self.switch_user(user_input.split(maxsplit=1)[1])
                    continue
                elif user_input == "/setup":
                    self.setup_profile()
                    continue
                elif user_input.startswith("/profile "):
                    self.set_profile_value(user_input.split(maxsplit=1)[1])
                    continue
                elif user_input == "/task":
                    self.show_task()
                    continue
                elif user_input.startswith("/task "):
                    self.create_task(user_input.split(maxsplit=1)[1])
                    continue
                elif user_input.startswith("/plan "):
                    self.set_task_plan(user_input.split(maxsplit=1)[1])
                    continue
                elif user_input.startswith("/work "):
                    self.set_memory_value("work", user_input.split(maxsplit=1)[1])
                    continue
                elif user_input.startswith("/work-forget "):
                    self.forget_memory_value("work", user_input.split(maxsplit=1)[1])
                    continue
                elif user_input == "/work-clear":
                    self.clear_working_memory()
                    continue
                elif user_input.startswith("/remember "):
                    self.set_memory_value("remember", user_input.split(maxsplit=1)[1])
                    continue
                elif user_input.startswith("/decision "):
                    self.set_memory_value("decision", user_input.split(maxsplit=1)[1])
                    continue
                elif user_input.startswith("/forget "):
                    self.forget_memory_value("long", user_input.split(maxsplit=1)[1])
                    continue
                elif user_input.startswith("/stage "):
                    self.transition_task(user_input.split(maxsplit=1)[1].strip())
                    continue
                elif user_input == "/invariants":
                    self.show_invariants()
                    continue
                elif user_input.startswith("/invariant "):
                    self.add_invariant(user_input.split(maxsplit=1)[1])
                    continue
                elif user_input == "/checkpoint":
                    self.create_checkpoint()
                    continue
                elif user_input == "/branches":
                    self.show_branches()
                    continue
                elif user_input.startswith("/branch "):
                    self.switch_branch(user_input.split(maxsplit=1)[1].strip())
                    continue
                elif user_input == "/mcp-start":
                    self.start_mcp_server()
                    continue
                elif user_input == "/mcp-tools":
                    self.show_mcp_tools()
                    continue
                elif user_input.startswith("/mcp-call "):
                    self.call_mcp_tool(user_input.split(maxsplit=1)[1])
                    continue
                elif user_input == "/mcp-stop":
                    self.mcp_client.stop()
                    print("\nMCP-сервер остановлен\n")
                    continue
                elif user_input == "/summary-now":
                    self.run_periodic_summary_once(force=True)
                    continue
                elif user_input == "/summary-status":
                    self.show_periodic_summary_status()
                    continue
                elif user_input == "/code-review-deep":
                    self.review_code_folder("", deep=True)
                    continue
                elif user_input.startswith("/code-review-deep "):
                    self.review_code_folder(user_input.split(maxsplit=1)[1], deep=True)
                    continue
                elif user_input.startswith("/code-review "):
                    self.review_code_folder(user_input.split(maxsplit=1)[1])
                    continue
                elif user_input == "/bugs-write":
                    self.write_bug_reports()
                    continue
                elif user_input == "/bugs-fix":
                    self.fix_bugs_from_folder()
                    continue
                elif user_input.startswith("/bugs-fix "):
                    self.fix_bugs_from_folder(user_input.split(maxsplit=1)[1])
                    continue
                elif self.is_natural_review_request(user_input):
                    self.review_code_folder(
                        self.extract_review_folder_from_text(user_input),
                        deep=self.is_deep_review_request(user_input),
                    )
                    continue
                elif self.is_natural_bug_fix_request(user_input):
                    self.fix_bugs_from_folder(
                        self.extract_optional_folder_from_text(user_input)
                    )
                    continue
                elif self.is_natural_bug_write_request(user_input):
                    self.write_bug_reports()
                    continue
                elif user_input == "/json":
                    use_json_mode = True
                    print("\nПереключено в режим JSON ответов\n")
                    continue
                elif user_input == "/normal":
                    use_json_mode = False
                    print("\nПереключено в обычный режим ответов\n")
                    continue
                elif not user_input:
                    continue

                temperature = self.get_temperature_from_user()
                print(f"Используется температура: {temperature}")

                if use_json_mode:
                    result = self.get_json_response(user_input, temperature)
                    if result:
                        print(f"\nОтвет: {result.get('answer', '')}")
                        print(f"Уверенность: {result.get('confidence', 0)}")
                        print(f"Интент: {result.get('intent', '')}")
                        if result.get("data"):
                            print(f"Данные: {json.dumps(result['data'], ensure_ascii=False)}")
                        self.print_last_token_counts()
                else:
                    if use_streaming:
                        self.send_message_streaming(user_input, temperature)
                    else:
                        self.send_message_simple(user_input, temperature)

                print()

            except KeyboardInterrupt:
                print("\n\nПрервано пользователем. До свидания!")
                break
            except Exception as e:
                print(f"\nНеожиданная ошибка: {e}")
                print("Попробуйте еще раз или введите /exit для выхода\n")

        self.stop_periodic_summary()
        self.mcp_client.stop()
        self.code_review_mcp_client.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Клиент для OpenAI-compatible локального LLM сервера."
    )
    parser.add_argument(
        "--server",
        help=(
            "Адрес сервера без пути, например 192.168.1.50:8080 "
            "или http://192.168.1.50:8080"
        ),
    )
    parser.add_argument(
        "--api-url",
        help=(
            "Полный URL chat completions endpoint, например "
            "http://192.168.1.50:8080/v1/chat/completions"
        ),
    )
    args = parser.parse_args()
    api_url = normalize_api_url(args.api_url or args.server or API_URL)

    print("Выберите режим работы:")
    print("1. Обычный режим (модель думает, затем выдает полный ответ)")
    print("2. Потоковый режим (ответ появляется по словам в реальном времени)")
    print("3. JSON режим (структурированные ответы)")

    choice = input("\nВаш выбор (1/2/3): ").strip()

    use_streaming = choice == "2"
    use_json_mode = choice == "3"

    print("\nВыберите стратегию контекста и памяти:")
    print("1. Только последние 10 сообщений")
    print("2. Facts key-value + последние 10 сообщений")
    print("3. Checkpoint и две независимые ветки диалога")

    strategy_choice = input("\nВаш выбор (1/2/3): ").strip()
    strategy_by_choice = {
        "1": CONTEXT_STRATEGY_RECENT,
        "2": CONTEXT_STRATEGY_FACTS,
        "3": CONTEXT_STRATEGY_BRANCHES,
    }
    context_strategy = strategy_by_choice.get(strategy_choice, CONTEXT_STRATEGY_RECENT)

    chat = LLMChat(api_url=api_url, context_strategy=context_strategy)
    chat.run(use_streaming=use_streaming, use_json_mode=use_json_mode)


if __name__ == "__main__":
    main()
