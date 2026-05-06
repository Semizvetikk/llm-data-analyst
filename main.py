import os
import re
import io
import sys
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class SafeCodeInterpreter:
    """
    Класс интерпретатора кода для LLM-агента.
    Изолированная среда для исполнения Python-кода, сгенерированного LLM.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.execution_history = []

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Выполняет Python-код в изолированном окружении.
        Возвращает результат или ошибку.
        """

        # Защита от опасных операций
        forbidden = [
            'import os', 'import sys', 'subprocess', 'pickle', 'shutil',
            '__import__', 'eval(', 'exec(', 'compile(', 'open(', 'input(',
            'os.', 'sys.', 'pty.', 'socket', 'http', 'urllib', 'requests'
        ]
        for pattern in forbidden:
            if pattern in code:
                return {"error": f"Запрещённая операция: {pattern}"}

        # Только разрешённые объекты. Ограничение по времени/итерациям реализуется на уровне LLM
        safe_scope = {
            "pd": pd, "df": self.df.copy(), "print": print,
            "len": len, "range": range, "list": list, "dict": dict,
            "sum": sum, "min": min, "max": max, "sorted": sorted,
            "float": float, "int": int, "str": str, "abs": abs,
            "round": round, "enumerate": enumerate, "zip": zip
        }

        # Перехват вывода print()
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            # Выполнение кода в изолированном окружении
            exec(code, {"__builtins__": {}}, safe_scope)
            output = captured.getvalue()

            result = {
                "status": "success",
                "output": output if output else "Код выполнен (нет вывода)",
                "data_sample": self._safe_sample(safe_scope) # Пример результата
            }
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        finally:
            sys.stdout = old_stdout

        self.execution_history.append({"code": code, "result": result})
        return result

    def _safe_sample(self, scope: dict) -> Optional[Dict]:
        """
        Возвращает пример результата для контекста LLM.

        param:
            scope - Словарь переменных после исполнения кода
        return: Краткое описание результата или None
        """
        try:
            for var_name in ["result", "out", "output", "ans"]:
                if var_name in scope and scope[var_name] is not None:
                    val = scope[var_name]
                    if isinstance(val, pd.DataFrame):
                        return {"type": "DataFrame", "head": val.head(3).to_dict(), "shape": val.shape}
                    elif isinstance(val, (pd.Series, list, dict)):
                        return {"type": type(val).__name__, "sample": str(val)[:200]}
                    else:
                        return {"type": type(val).__name__, "value": str(val)[:200]}
        except:
            pass
        return None


def sanitize_input(text: str, max_length: int = 3000) -> str:
    """Защита ввода от prompt-injection и опасных символов"""
    patterns = [
        r'ignore\s+(previous|all|system)', r'forget\s+instructions',
        r'disregard\s+prompt', r'jailbreak', r'bypass\s+filter',
        r'system\s*(prompt|instruction)', r'role:\s*(system|assistant)',
        r'ты теперь', r'игнорируй', r'обойди защиту', r'act as developer'
    ]
    text_lower = text.lower()
    for p in patterns:
        # Если найден запрещённый паттерн - блокируем запрос
        if re.search(p, text_lower, re.I):
            raise ValueError("Обнаружена попытка инъекции")

    # Проверка на превышение длины
    if len(text) > max_length:
        raise ValueError(f"Превышен лимит длины ({max_length} символов)")

    # Удаление потенциально опасных символов
    for ch in ['<', '>', '|', '`', ';', '&', '$', '#', '!', '\\']:
        text = text.replace(ch, '')
    return text.strip()


def extract_code_blocks(text: str) -> list[str]:
    """Извлекает код из ответа LLM (между ```python и ```)"""
    pattern = r'```(?:python)?\s*\n(.*?)```'
    return [block.strip() for block in re.findall(pattern, text, re.DOTALL)]


def run_agent_analysis(
        df: pd.DataFrame,
        user_query: str, # Вопрос пользователя к данным
        system_prompt: str, # Инструкции для LLM (редактируемые)
        max_iterations: int = 8
) -> str:
    """Запуск LLM-агента с циклом Code Interpreter."""
    try:
        user_query = sanitize_input(user_query)
    except ValueError as e:
        return f"Безопасность: {e}"

    interpreter = SafeCodeInterpreter(df)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Датасет загружен. Запрос: {user_query}"}
    ]

    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=messages,
                temperature=0.1,
                timeout=90
            )

            assistant_content = response.choices[0].message.content
            code_blocks = extract_code_blocks(assistant_content)

            # Если кода нет - LLM сформировала финальный текстовый ответ
            if not code_blocks:
                return assistant_content

            # Исполняем каждый блок кода и собираем результаты в контекст
            tool_results = []
            for i, code in enumerate(code_blocks, 1):
                exec_result = interpreter.execute(code)
                tool_results.append(
                    f"### Код {i}:\n```python\n{code}\n```\n### Результат:\n{json.dumps(exec_result, ensure_ascii=False, indent=2)}")

            # Добавляем в диалог: ответ ассистента + результаты исполнения
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": f"Результаты исполнения кода (итерация {iteration + 1}):\n\n" + "\n\n".join(tool_results)
            })

        except Exception as e:
            return f"Ошибка на итерации {iteration + 1}: {str(e)}"

    return "Достигнут лимит итераций. Анализ прерван."


# Интерфейс STREAMLIT
st.set_page_config(page_title="LLM Data Analyst", layout="wide")
st.title("LLM-агент для анализа данных")

# Боковая панель с настройками
with st.sidebar:
    st.header("Настройки")

    # Системный промпт
    default_prompt = """Ты анализируешь данные через Python-интерпретатор.
    Данные в переменной `df` (pandas DataFrame). 
    Используй print() для вывода. Отвечай по сути.
    """

    system_prompt = st.text_area(
        "Системный промпт",
        value=default_prompt,
        height=300,
        help="Инструкции для LLM. Можно редактировать под задачу."
    )

    st.info("Подсказка: в коде используй `print(df.head())`, `print(df['col'].mean())` и т.д.")

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

# Поле для вопроса пользователя (пустое по умолчанию, с placeholder-подсказкой)
user_query = st.text_area(
    "Ваш вопрос к данным",
    value="",
    height=80,
    placeholder="Например: найди корреляции и выдели 3 инсайта..."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Быстрая сводка по датасету
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Строк", df.shape[0])
        col2.metric("Колонки", df.shape[1])
        col3.metric("Пропуски", df.isnull().sum().sum())
        col4.metric("Память", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        with st.expander("Просмотр данных", expanded=False):
            st.dataframe(df.head(10))
            st.write("Типы данных:", df.dtypes.to_dict())

        # Кнопка запуска анализа
        if st.button("Запустить анализ", type="primary", use_container_width=True):
            with st.spinner("LLM-агент работает..."):
                result = run_agent_analysis(df, user_query, system_prompt)
                st.markdown(result)

    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
else:
    st.info("Загрузите CSV-файл, чтобы начать анализ")
