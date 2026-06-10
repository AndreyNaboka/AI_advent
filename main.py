#!/usr/bin/env python3
import requests
import json
import time
from pathlib import Path
from threading import Thread
from typing import List, Dict, Optional, Any, TypedDict, Literal
from jsonschema import validate, ValidationError

API_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_TEMPERATURE = 0.7
HISTORY_FILE = Path(__file__).with_name("conversation_history.json")


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

    def __init__(self, api_url: str = API_URL, history_file: Path = HISTORY_FILE):
        self.api_url: str = api_url
        self.history_file = history_file
        self.conversation_history: List[Dict[str, str]] = self.load_history()
        self.running: bool = True

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

    def save_history(self) -> None:
        """Сохраняет историю диалога в файл."""
        try:
            with self.history_file.open("w", encoding="utf-8") as file:
                json.dump(
                    self.conversation_history,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError as e:
            print(f"Не удалось сохранить историю диалога: {e}")

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
        """Получает ответ в JSON формате с валидацией по схеме, показывает время и токены"""
        schema_to_use = schema or self.RESPONSE_SCHEMA

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

        user_message = (
            f"Вопрос: {message}\n\nВерни ответ строго в указанном JSON формате."
        )

        self.conversation_history.append({"role": "user", "content": message})

        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}]
            + self.conversation_history[:-1]
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

            # Получаем информацию о токенах
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            print(
                f"\nJSON ответ: {json.dumps(parsed_json, indent=2, ensure_ascii=False)}"
            )
            print(f"⏱️ Время ответа: {elapsed_time:.2f} сек")
            print(f"🔢 Токены: запрос {prompt_tokens}, ответ {completion_tokens}, всего {total_tokens}")

            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(parsed_json, ensure_ascii=False),
                }
            )
            self.save_history()

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
        """Отправляет сообщение с потоковой передачей ответа, показывает время и примерную длину"""

        self.conversation_history.append({"role": "user", "content": message})

        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}]
            + self.conversation_history,
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

            # Приблизительный подсчёт токенов (среднее: 1 токен ~= 3 символа для русского/английского)
            approx_tokens = len(full_response) // 3
            print(f"⏱️ Время ответа: {elapsed_time:.2f} сек")
            print(f"📝 Примерная длина: {len(full_response)} символов (~{approx_tokens} токенов)")

            self.conversation_history.append({"role": "assistant", "content": full_response})
            self.save_history()
            return full_response

        except Exception as e:
            print(f"\nОшибка: {e}")
            return None

    def send_message_simple(
        self, message: str, temperature: float, system_prompt: str = "Ты полезный ассистент."
    ) -> Optional[str]:
        """Отправляет сообщение без потокового вывода, показывает время и токены"""
        self.conversation_history.append({"role": "user", "content": message})

        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}]
            + self.conversation_history,
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
            print(f"\nАссистент: {assistant_message}")

            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            print(f"⏱️ Время ответа: {elapsed_time:.2f} сек")
            print(f"🔢 Токены: запрос {prompt_tokens}, ответ {completion_tokens}, всего {total_tokens}")

            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            self.save_history()
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
        self.conversation_history = []
        try:
            self.history_file.unlink(missing_ok=True)
        except OSError as e:
            print(f"История очищена в памяти, но файл удалить не удалось: {e}")
            return
        print("История диалога очищена")

    def show_help(self) -> None:
        print("\n" + "=" * 50)
        print("Команды:")
        print("  /clear   - Очистить историю диалога")
        print("  /history - Показать историю диалога")
        print("  /status  - Проверить статус сервера")
        print("  /json    - Режим JSON ответов")
        print("  /normal  - Обычный режим ответов")
        print("  /help    - Показать эту справку")
        print("  /exit    - Выйти из программы")
        print("=" * 50 + "\n")

    def show_history(self) -> None:
        if not self.conversation_history:
            print("\nИстория пуста\n")
            return
        print("\n" + "=" * 50)
        print("ИСТОРИЯ ДИАЛОГА:")
        print("=" * 50)
        for i, msg in enumerate(self.conversation_history, 1):
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

    def run(self, use_streaming: bool = False, use_json_mode: bool = False) -> None:
        print("\n" + "=" * 50)
        print("ДОБРО ПОЖАЛОВАТЬ В ЧАТ С LOCAL LLM")
        print("=" * 50)
        print(f"Модель: Qwen3-4B-Instruct (локальная)")
        print(f"Режим: {'потоковый' if use_streaming else 'обычный'}")
        print(f"Формат ответа: {'JSON' if use_json_mode else 'текстовый'}")
        print("Введите /help для списка команд")
        print(f"Файл истории: {self.history_file}")
        if self.conversation_history:
            print(f"Загружено сообщений из истории: {len(self.conversation_history)}")
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
                elif user_input == "/help":
                    self.show_help()
                    continue
                elif user_input == "/status":
                    self.check_status()
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
    chat = LLMChat()
    print("Выберите режим работы:")
    print("1. Обычный режим (модель думает, затем выдает полный ответ)")
    print("2. Потоковый режим (ответ появляется по словам в реальном времени)")
    print("3. JSON режим (структурированные ответы)")

    choice = input("\nВаш выбор (1/2/3): ").strip()

    use_streaming = choice == "2"
    use_json_mode = choice == "3"

    chat.run(use_streaming=use_streaming, use_json_mode=use_json_mode)


if __name__ == "__main__":
    main()
