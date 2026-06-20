#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import List, Dict, Optional, Any, TypedDict, Literal


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
from jsonschema import validate, ValidationError

API_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_TEMPERATURE = 0.7
HISTORY_FILE = Path(__file__).with_name("conversation_history.json")
SUMMARY_FILE = Path(__file__).with_name("conversation_summary.json")
FACTS_FILE = Path(__file__).with_name("conversation_facts.json")
BRANCHES_FILE = Path(__file__).with_name("conversation_branches.json")
PROFILE_FILE = Path(__file__).with_name("user_profile.json")
TASK_STATE_FILE = Path(__file__).with_name("task_state.json")
INVARIANTS_FILE = Path(__file__).with_name("invariants.json")
MEMORY_FILE = Path(__file__).with_name("agent_memory.json")
RECENT_MESSAGES_LIMIT = 10
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
        context_strategy: str = CONTEXT_STRATEGY_RECENT,
    ):
        self.api_url: str = api_url
        self.tokenize_url: str = api_url.split("/v1/", 1)[0] + "/tokenize"
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
        self.running: bool = True
        self.tokenize_available: Optional[bool] = None
        self.last_token_counts: Optional[Dict[str, int]] = None
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
            matched = [
                term for term in invariant.get("forbidden_terms", [])
                if term and term.casefold() in normalized
            ]
            if matched:
                violations.append(
                    f"{invariant['rule']} (найдено: {', '.join(matched)})"
                )
        return violations

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
        except:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("localhost", 8080))
                sock.close()
                return result == 0
            except:
                return False

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
                    print("\nПредупреждение: ответ всё ещё нарушает инварианты: " + "; ".join(remaining))

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
        self, message: str, temperature: float, system_prompt: str = "Ты полезный ассистент."
    ) -> Optional[str]:
        """Отправляет сообщение без потокового вывода, показывает время и токены"""
        self.append_message("user", message)
        self.trim_active_history_if_needed()
        self.update_facts_from_user_message(message)

        stateful_prompt = self.build_stateful_system_prompt(system_prompt)
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
                    print("\nПредупреждение: ответ всё ещё нарушает инварианты: " + "; ".join(remaining))
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
            print("   python3 -m llama_cpp.server --model ~/models/qwen3-4b/qwen3-4b-instruct-2507-q8_0.gguf --n_gpu_layers 99 --port 8080")
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
            print("Сервер доступен (порт 8080)")
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
            print("   python3 -m llama_cpp.server --model ~/models/qwen3-4b/qwen3-4b-instruct-2507-q8_0.gguf --n_gpu_layers 99 --port 8080")
        print()

    def show_facts(self) -> None:
        if not self.facts:
            print("\nДолговременные знания пусты\n")
            return
        print("\nДОЛГОВРЕМЕННЫЕ ЗНАНИЯ:")
        print(json.dumps(self.facts, ensure_ascii=False, indent=2))
        print()

    def run(self, use_streaming: bool = False, use_json_mode: bool = False) -> None:
        print("\n" + "=" * 50)
        print("ДОБРО ПОЖАЛОВАТЬ В ЧАТ С LOCAL LLM")
        print("=" * 50)
        print(f"Модель: Qwen3-4B-Instruct (локальная)")
        print(f"Режим: {'потоковый' if use_streaming else 'обычный'}")
        print(f"Формат ответа: {'JSON' if use_json_mode else 'текстовый'}")
        print(f"Стратегия контекста: {self.context_strategy}")
        print(f"Активный пользователь: {self.current_user}")
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
            print("cd ~/llm_project && source venv/bin/activate")
            print("python3 -m llama_cpp.server --model ~/models/qwen3-4b/qwen3-4b-instruct-2507-q8_0.gguf --n_gpu_layers 99 --port 8080")
            print("\nПосле запуска сервера, перезапустите эту программу.")
            return

        print("Сервер доступен! Готов к работе.\n")

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


def main() -> None:
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

    chat = LLMChat(context_strategy=context_strategy)
    chat.run(use_streaming=use_streaming, use_json_mode=use_json_mode)


if __name__ == "__main__":
    main()
