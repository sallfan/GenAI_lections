# llm_agent/streaming_client.py
import json
import re
import requests
from typing import Dict, Optional, Callable, Iterator, Union, List

from .core_v2 import LLMAgent


class StreamingLLMClient(LLMAgent):
    """
    Расширение LLMAgent с поддержкой потоковой передачи токенов (streaming)
    и пошагового вывода промежуточных мыслей.
    """

    def _make_api_request(
        self,
        payload: Dict,
        headers: Optional[Dict] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Union[Dict, Iterator[str]]:

        if not stream:
            return super()._make_api_request(payload, headers)
        return self._make_streaming_request(payload, headers, on_token)

    def _make_streaming_request(
        self,
        payload: Dict,
        headers: Optional[Dict] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[str], None]] = None,
        show_thinking: bool = False,
    ) -> Iterator[str]:
        if headers is None:
            headers = {}

        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"

        stream_payload = dict(payload)
        stream_payload["stream"] = True

        try:
            response = requests.post(self.url, json=stream_payload, headers=headers, stream=True, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")

        in_think = False
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8")
            else:
                line = raw_line

            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if not data_str:
                continue
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            try:
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or choice.get("reasoning_content")
                if reasoning:
                    if show_thinking:
                        notify = on_step if on_step is not None else (lambda m: print(f"[thinking] {m}", flush=True))
                        try:
                            notify(reasoning)
                        except Exception:
                            pass
                    else:
                        pass
                    if not delta.get("content") and not choice.get("message", {}).get("content"):
                        continue

                token = delta.get("content")
                if token is None:
                    token = choice.get("message", {}).get("content")
                if token is None:
                    token = chunk.get("text", "")
                    if not token:
                        continue

                if token:
                    if in_think:
                        if "</think>" in token:
                            think_part, rest = token.split("</think>", 1)
                            if think_part.strip() and show_thinking:
                                notify = on_step if on_step is not None else (lambda m: print(f"[thinking] {m}", flush=True))
                                try:
                                    notify(think_part.strip())
                                except Exception:
                                    pass
                            token = rest
                            in_think = False
                            if not token.strip():
                                continue
                        else:
                            if token.strip() and show_thinking:
                                notify = on_step if on_step is not None else (lambda m: print(f"[thinking] {m}", flush=True))
                                try:
                                    notify(token.strip())
                                except Exception:
                                    pass
                            continue
                    elif "<think>" in token:
                        if "</think>" in token:
                            thinks = re.findall(r"<think>(.*?)</think>", token, re.DOTALL)
                            for th in thinks:
                                th = th.strip()
                                if th and show_thinking:
                                    notify = on_step if on_step is not None else (lambda m: print(f"[thinking] {m}", flush=True))
                                    try:
                                        notify(th)
                                    except Exception:
                                        pass
                            token = re.sub(r"<think>.*?</think>", "", token, flags=re.DOTALL)
                            if not token.strip():
                                continue
                        else:
                            # незакрытый <think> — начало блока
                            before, after = token.split("<think>", 1)
                            if after.strip() and show_thinking:
                                notify = on_step if on_step is not None else (lambda m: print(f"[thinking] {m}", flush=True))
                                try:
                                    notify(after.strip())
                                except Exception:
                                    pass
                            token = before
                            in_think = True
                            if not token.strip():
                                continue

                    if on_token is not None:
                        try:
                            on_token(token)
                        except Exception:
                            pass
                    yield token
            except (IndexError, AttributeError, KeyError):
                continue

    def _ask_llm_for_plan(
        self,
        query: str,
        show_thinking: bool = False,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> List[Dict]:
        system_prompt = f"""
        You are a helpful AI planning assistant. Analyze the user's request and decide if you need to use any tools.
        Available tools:
        - **calculator**: For any math-related questions (numbers, calculations). Use it with the full expression.
        - **web_search**: For finding any information about the real world (current events, facts, definitions). Use it with the user's question or a clear search query. USE ONLY RUSSIAN LANGUAGE QUERIES in this tool.
        - **pdf_info**: For extracting information from PDF files (metadata, page count, text content). Use it with a local file path or a URL to a PDF file.
        Your response MUST be ONLY a JSON object of the following format.
        If one or more tools are needed to answer, return JSON of this structure:
        {{
        "plan": [
            {{"action": "tool_name", "input": "some text to pass into tool"}},
            ... //MORE ACTIONS IF NEEDED SEVERAL TOOLS. ONE ACTION FOR ONE TOOL CALL
        ]
        }}
        If no tool is needed, return an empty plan: {{"plan": []}}.
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }

        llm_text = ""
        try:
            if show_thinking:
                # Стрим с захватом thinking через on_step — не ограничиваем stream
                payload.pop("stream", None)
                buf: List[str] = []
                for tok in self._make_streaming_request(payload, on_token=None, on_step=on_step, show_thinking=True):
                    buf.append(tok)
                llm_text = "".join(buf)
            else:
                # Совместимый non-stream, но с очисткой thinking
                if self.local:
                    payload["stream"] = False
                response_data = self._make_api_request(payload, stream=False)
                # Ollama non-stream может отдавать reasoning отдельно
                choice = response_data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                llm_text = msg.get("content", "")
                # если reasoning в отдельном поле — показываем только при show_thinking, иначе игнор
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                if reasoning and show_thinking and on_step is not None:
                    try:
                        on_step(reasoning)
                    except Exception:
                        pass

            # Очистка thinking/reasoning из текста перед парсингом JSON
            # Удаляем <think> блоки полностью
            cleaned_for_json = re.sub(r"<think>.*?</think>", "", llm_text, flags=re.DOTALL)
            cleaned_for_json = re.sub(r"</?think>", "", cleaned_for_json)
            # Убираем markdown обёртки
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_for_json, re.DOTALL)
            if json_match:
                cleaned_json_text = json_match.group(1)
            else:
                cleaned_json_text = cleaned_for_json

            if show_thinking:
                notify = on_step if on_step is not None else (lambda m: print(f"[thinking] {m}", flush=True))
                try:
                    # логируем очищенный план для дебага, но не спамим
                    pass
                except Exception:
                    pass

            action_plan = json.loads(cleaned_json_text)
            plan = action_plan.get("plan", [])
            return plan

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Произошла ошибка при создании плана: {e}")
            try:
                # fallback: ищем JSON без маркеров в очищенном тексте
                cleaned_fallback = re.sub(r"<think>.*?</think>", "", llm_text, flags=re.DOTALL)
                cleaned_fallback = re.sub(r"</?think>", "", cleaned_fallback)
                json_match = re.search(r'\{.*"plan".*\}', cleaned_fallback, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group())
                    return action_plan.get("plan", [])
            except Exception:
                pass
            return []

    def stream_generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        yield from self._make_api_request(payload, stream=True, on_token=on_token)

    def process_query(
        self,
        query: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[str], None]] = None,
        show_thinking: bool = False,
        stream: Optional[bool] = None,
    ) -> str:
        """
            query: запрос пользователя
            on_token: callback(str) для каждого токена финального ответа (пошаговый вывод)
            on_step: callback(str) для промежуточных шагов (план, инструменты, thinking)
            show_thinking: если True — выводить reasoning / <think> через on_step
            stream: явный флаг; если None — auto (True когда on_token задан)
        """
        use_stream = stream if stream is not None else (on_token is not None)

        def _notify(msg: str):
            if on_step is not None:
                try:
                    on_step(msg)
                except Exception:
                    print(msg, flush=True)
            else:
                print(msg, flush=True)

        _notify(f"Агент анализирует запрос... (Режим: {'локальный Ollama' if self.local else 'OpenRouter'})")

        if show_thinking:
            _notify("\n[thinking] Формирую план действий...\n")
        plan = self._ask_llm_for_plan(query, show_thinking=show_thinking, on_step=on_step)

        if not plan:
            _notify("\nИнструменты не требуются. Генерирую ответ напрямую.")
            direct_prompt = f"Ответьте на следующий вопрос кратко и информативно: {query}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": direct_prompt}],
            }
            if self.local:
                payload.pop("stream", None)
            if not use_stream:
                try:
                    response_data = self._make_api_request(payload, stream=False)
                    return response_data["choices"][0]["message"]["content"]
                except Exception:
                    return "Извините, не удалось сгенерировать ответ."
            tokens: List[str] = []
            for tok in self._make_streaming_request(payload, on_token=None, on_step=on_step, show_thinking=show_thinking):
                tokens.append(tok)
                if on_token is not None:
                    try:
                        on_token(tok)
                    except Exception:
                        print(tok, end="", flush=True)
                else:
                    print(tok, end="", flush=True)
            if on_token is None:
                print()
            return "".join(tokens)

        _notify(f"\nПлан действий: {plan}\n")
        for step in plan:
            tool_name = step.get("action")
            tool_input = step.get("input")
            if tool_name in self.tools:
                _notify(f"\nВыполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                _notify(f"\nРезультат: {result[:200]}...")
                self.conversation_history.append({
                    "role": "system",
                    "content": f"Tool {tool_name} result: {result}"
                })
            else:
                error_msg = f"\nОшибка: инструмент '{tool_name}' не найден."
                _notify(error_msg)
                self.conversation_history.append({"role": "system", "content": error_msg})

        _notify("\nСоставляю финальный ответ...\n\n")
        prompt = f"""
        Based on the following conversation log, provide a direct and helpful answer to the user's original question.
        Be concise and use the information from the tool results to support your answer.

        Original User Question: {query}

        Conversation Log:
        {chr(10).join([msg['content'] for msg in self.conversation_history])}
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not use_stream:
            return self._generate_final_response(query)

        tokens = []
        for tok in self._make_streaming_request(payload, on_token=None, on_step=on_step, show_thinking=show_thinking):
            tokens.append(tok)
            if on_token is not None:
                try:
                    on_token(tok)
                except Exception:
                    print(tok, end="", flush=True)
            else:
                print(tok, end="", flush=True)
        if on_token is None:
            print()
        return "".join(tokens)