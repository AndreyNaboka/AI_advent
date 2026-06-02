#!/usr/bin/env python3
import requests
import json
import time
from threading import Thread
from typing import List, Dict, Optional, Any

API_URL = "http://localhost:8080/v1/chat/completions"


class LLMChat:
    def __init__(self, api_url: str = API_URL):
        self.api_url: str = api_url
        self.conversation_history: List[Dict[str, str]] = []
        self.running: bool = True

    def check_server(self) -> bool:
        """Проверяет, доступен ли сервер"""
        try:
            # Простая проверка через options запрос
            requests.options(self.api_url, timeout=2)
            return True
        except:
            # Альтернативная проверка - пробуем подключиться к порту
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("localhost", 8080))
                sock.close()
                return result == 0
            except:
                return False

    def send_with_progress(self, payload: Dict[str, Any]) -> requests.Response:
        """Отправляет запрос с индикатором прогресса"""
        print("Модель думает", end="", flush=True)

        # Запускаем индикатор прогресса в отдельном потоке
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
            print("\r" + " " * 30 + "\r", end="", flush=True)  # Очищаем строку
            return response
        except Exception as e:
            stop_animation = True
            progress_thread.join(timeout=0.5)
            print("\r" + " " * 30 + "\r", end="", flush=True)
            raise e

    def send_message_streaming(
        self, message: str, system_prompt: str = "Ты полезный ассистент."
    ) -> Optional[str]:
        """Отправляет сообщение с потоковой передачей ответа (постепенный вывод)"""

        # Добавляем сообщение в историю
        self.conversation_history.append({"role": "user", "content": message})

        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}]
            + self.conversation_history,
            "max_tokens": 1000,
            "temperature": 0.7,
            "stream": True,  # Включаем потоковый режим
        }

        print("\nАссистент: ", end="", flush=True)

        full_response: str = ""
        try:
            # Для потокового режима нужен отдельный эндпоинт или обработка
            # В llama.cpp сервере потоковый режим работает через событийный поток
            response: requests.Response = requests.post(
                self.api_url, json=payload, stream=True, timeout=60
            )

            for line in response.iter_lines():
                if line:
                    line_str: str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data: str = line_str[6:]
                        if data != "[DONE]":
                            try:
                                chunk: Dict[str, Any] = json.loads(data)
                                if "choices" in chunk and chunk["choices"][0].get(
                                    "delta", {}
                                ).get("content"):
                                    content: str = chunk["choices"][0]["delta"][
                                        "content"
                                    ]
                                    print(content, end="", flush=True)
                                    full_response += content
                            except:
                                pass

            print()  # Новая строка после ответа
            self.conversation_history.append(
                {"role": "assistant", "content": full_response}
            )
            return full_response

        except Exception as e:
            print(f"\nОшибка: {e}")
            return None

    def send_message_simple(
        self, message: str, system_prompt: str = "Ты полезный ассистент."
    ) -> Optional[str]:
        """Отправляет сообщение без потокового вывода"""

        # Добавляем сообщение в историю
        self.conversation_history.append({"role": "user", "content": message})

        payload: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}]
            + self.conversation_history,
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        try:
            response: requests.Response = self.send_with_progress(payload)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()

            assistant_message: str = result["choices"][0]["message"]["content"]
            print(f"\nАссистент: {assistant_message}")

            # Сохраняем ответ в историю
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )
            return assistant_message

        except requests.exceptions.Timeout:
            print("\nОшибка: Превышено время ожидания ответа от модели")
            return None
        except requests.exceptions.ConnectionError:
            print(
                "\nОшибка: Не удалось подключиться к серверу. Запустите сервер командой:"
            )
            print(
                "   python3 -m llama_cpp.server --model ~/models/qwen3-4b/qwen3-4b-instruct-2507-q8_0.gguf --n_gpu_layers 99 --port 8080"
            )
            return None
        except Exception as e:
            print(f"\nОшибка: {e}")
            return None

    def clear_history(self) -> None:
        """Очищает историю диалога"""
        self.conversation_history = []
        print("История диалога очищена")

    def show_help(self) -> None:
        """Показывает справку"""
        print("\n" + "=" * 50)
        print("Команды:")
        print("  /clear  - Очистить историю диалога")
        print("  /history - Показать историю диалога")
        print("  /status - Проверить статус сервера")
        print("  /help   - Показать эту справку")
        print("  /exit или /quit - Выйти из программы")
        print("=" * 50 + "\n")

    def show_history(self) -> None:
        """Показывает историю диалога"""
        if not self.conversation_history:
            print("\nИстория пуста\n")
            return

        print("\n" + "=" * 50)
        print("ИСТОРИЯ ДИАЛОГА:")
        print("=" * 50)
        for i, msg in enumerate(self.conversation_history, 1):
            role_label: str = "Пользователь" if msg["role"] == "user" else "Ассистент"
            print(
                f"{role_label}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}"
            )
        print("=" * 50 + "\n")

    def check_status(self) -> None:
        """Проверяет статус сервера"""
        print("\nПроверка статуса...")

        if self.check_server():
            print("Сервер доступен (порт 8080)")

            # Дополнительная проверка - пробный запрос
            try:
                test_payload: Dict[str, Any] = {
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                }
                response: requests.Response = requests.post(
                    self.api_url, json=test_payload, timeout=5
                )
                if response.status_code == 200:
                    print("Модель отвечает на запросы")
                else:
                    print(f"Сервер ответил с кодом: {response.status_code}")
            except:
                print("Модель не отвечает на тестовый запрос")
        else:
            print("Сервер недоступен! Запустите сервер командой:")
            print(
                "   python3 -m llama_cpp.server --model ~/models/qwen3-4b/qwen3-4b-instruct-2507-q8_0.gguf --n_gpu_layers 99 --port 8080"
            )

        print()

    def run(self, use_streaming: bool = False) -> None:
        """Запускает основной цикл чата"""
        print("\n" + "=" * 50)
        print("ДОБРО ПОЖАЛОВАТЬ В ЧАТ С LOCAL LLM")
        print("=" * 50)
        print(f"Модель: Qwen3-4B-Instruct (локальная)")
        print(f"Режим: {'потоковый' if use_streaming else 'обычный'}")
        print("Введите /help для списка команд")
        print("=" * 50 + "\n")

        # Проверяем доступность сервера при старте
        if not self.check_server():
            print("СЕРВЕР НЕ ДОСТУПЕН!")
            print("\nЗапустите сервер в другом терминале:")
            print("cd ~/llm_project && source venv/bin/activate")
            print(
                "python3 -m llama_cpp.server --model ~/models/qwen3-4b/qwen3-4b-instruct-2507-q8_0.gguf --n_gpu_layers 99 --port 8080"
            )
            print("\nПосле запуска сервера, перезапустите эту программу.")
            return

        print("Сервер доступен! Готов к работе.\n")

        while self.running:
            try:
                # Получаем ввод пользователя
                user_input: str = input("Вы: ").strip()

                # Проверяем команды
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
                elif not user_input:
                    continue

                # Отправляем сообщение модели
                if use_streaming:
                    self.send_message_streaming(user_input)
                else:
                    self.send_message_simple(user_input)

                print()  # Пустая строка для разделения сообщений

            except KeyboardInterrupt:
                print("\n\nПрервано пользователем. До свидания!")
                break
            except Exception as e:
                print(f"\nНеожиданная ошибка: {e}")
                print("Попробуйте еще раз или введите /exit для выхода\n")


def main() -> None:
    # Создаем экземпляр чата
    chat = LLMChat()

    # Выбираем режим работы
    print("Выберите режим работы:")
    print("1. Обычный режим (модель думает, затем выдает полный ответ)")
    print("2. Потоковый режим (ответ появляется по словам в реальном времени)")

    choice: str = input("\nВаш выбор (1/2): ").strip()

    use_streaming: bool = choice == "2"

    # Запускаем чат
    chat.run(use_streaming=use_streaming)


if __name__ == "__main__":
    main()
