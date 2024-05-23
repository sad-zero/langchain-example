"""
langchain Agent Example
"""

import logging
from operator import itemgetter
from pprint import pprint
from typing import Dict, Literal
from langchain_community.chat_models.ollama import ChatOllama
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_react_agent
from langchain import hub

MODEL_NAME = "phi3:3.8b-mini-instruct-4k-q4_K_M"
agent_defined_functions = {}
llm = ChatOllama(model=MODEL_NAME)


@tool("calculate mathematical expression")
def calculate_mathematical_expression(function_call_expression: str) -> int | float | bool:
    """
    Calculate the mathematical expression with python and return the result.
    If the given expression has user defined functions, you should define them by using the "define mathematical function" tool.
    If the answer is False, you should rewrite the expression as python function call.

    [Example]
    expression: fibonacci(10)
    answer: 55

    expression:
    def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

    result = fibonacci(10)
    print(result)

    answer: False

    <format>
    @param expression: Python function call for mathematical expression. The expression should be executed by eval() function like eval(expression)
    @return The result of calculating the expression. If eval(expression) is failed, False is returned
    </format>
    """
    logging.debug(f"Enter {__name__}")
    # llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    # return llm_math_chain.invoke(expression)
    try:
        result = eval(function_call_expression)
    except:
        return False
    else:
        return result


@tool("register mathematical python function")
def register_mathematical_python_function_in_globals(python_function_definition: str) -> bool:
    """
    Register a python function from the given mathematical function in the global context and return whether the definition successes or fails
    <format>
    @param func: mathematical function expression. The function format should be "def func_name: ...".
    @return True if registration is successed else False
    </format>
    """
    logging.debug(f"Enter {__name__}")
    try:
        python_function_definition = python_function_definition.replace("```python", "")
        python_function_definition = python_function_definition.replace("```", "").strip()
        exec(python_function_definition, globals())
        exec(python_function_definition, agent_defined_functions)
    except:
        return False
    else:
        return True


@tool("format mathematical function definition to python function definition")
def format_mathematical_function_definition_to_python_function_definition(func: str) -> str:
    """
    Format the given function definition to use `eval()` function in python and return the formatted function definition
    <format>
    @param func: Invalid format python function definition
    @return Well-formatted python function definition
    </format>
    """
    logging.debug(f"Enter {__name__}")
    persona = "You are python syntax parser. you can make well-formatted python code from invalid-formatted python code"
    format_ = """
[Input/Output Format]
Input: Invalid format python function definition
Output: Well-formatted python function definition
""".strip()
    examples = """
[Example]
[Input]
def fibonnaci(n): if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2)
[Answer]
def fibonnaci(n):
  if n <= 1:
    return n
  else:
    return fibonacci(n-1) + fibonacci(n-2)
""".strip()

    system_prompt = "\n\n".join([persona, format_, examples])

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(system_prompt),
            HumanMessage("{query}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": func})


@tool("search")
def search(query: str) -> Dict[Literal["answer"], str]:
    """
    Search the knowledge base for the query
    <format>
    @param query: The question that user want to search
    @return The result of searching the query
    </format>
    """
    logging.debug(f"Enter {__name__}")
    persona = "You are very powerful search engine."
    format_ = """
[Input/Output Format]
Input: Natural string query.
Output: Wrapped by json format like {answer: {the answer of query}}
""".strip()
    context = """
[Context]
Users're asking mathmetical questions. you should be answer the matched formular for the question
""".strip()
    examples = """
[Example]
Q1. What is the fibbonaci's formular?
A1. fibo(n) = fibo(n-1) + fibo(n-2) (n > 1); fibo(0) = 0; fibo(1) = 1;

Q2. What is the area of circle's formular?
A2. area(r) = r*r*pi; pi := math.pi;
""".strip()

    system_prompt = "\n\n".join([persona, format_, context, examples])

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(system_prompt),
            HumanMessage("{query}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})


if __name__ == "__main__":
    prompt = hub.pull("hwchase17/react")
    tools1 = [
        register_mathematical_python_function_in_globals,
        format_mathematical_function_definition_to_python_function_definition,
    ]
    agent1 = create_react_agent(llm, tools1, prompt)
    agent_executor1 = AgentExecutor(
        agent=agent1,
        tools=tools1,
        max_iterations=10,
        return_intermediate_steps=True,
        handle_parsing_errors="Think step by step",
        verbose=True,
    )

    tools2 = [calculate_mathematical_expression]
    agent2 = create_react_agent(llm, tools2, prompt)
    agent_executor2 = AgentExecutor(
        agent=agent2,
        tools=tools2,
        max_iterations=10,
        return_intermediate_steps=True,
        handle_parsing_errors="Think step by step",
        verbose=True,
    )

    query1 = "What is the fibonacci? Register the fibonnaci function and return the function name"
    query2 = "Using defined function, {func}, calculate {func}(13)"
    # print(agent_executor.agent.llm_chain.prompt.template)
    answer1 = agent_executor1.invoke({"input": query1})
    pprint(answer1)
    # print("=" * 20)
    # answer2 = agent_executor2.invoke({"input": query2.format(func=answer1["output"])})
    # pprint(answer2)
    breakpoint()

    # 할루시네이션이 심각하다.
    # 한번에 하나의 일만 하도록 만들어야 한다. 두 개의 일을 하도록 시키니 계속 Action을 선택하려고 하는 오류가 발생한다.
    # 어떤 프롬프트를 사용하는지에 따라 성능이 좌우된다.
    # handle_parsing_errors를 어떻게 처리할지 중요하다.
