#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import matplotlib.pyplot as plt
import networkx as nx


# In[2]:


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# In[3]:



openai_api_key = ""

tavily_api_key = ""

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
tavily_client = TavilyClient(api_key=tavily_api_key)


# In[4]:


PLAN_PROMPT = """You are an expert writer tasked with writing a high-level outline of an essay.
Write such an outline for the user-provided topic. Give an outline of the essay along with any relevant notes or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.
Generate the best essay possible for the user-provided topic using the outline. Write a compelling introduction, three body paragraphs, and a conclusion."""

CRITIQUE_PROMPT = """You are an expert critic tasked with improving essays. Critique the following essay and suggest improvements."""

REVISION_PROMPT = """You are an essay assistant tasked with revising essays based on the following critique. Implement the suggested improvements and provide a revised version of the essay."""


# In[5]:


def create_plan(topic: str) -> str:
    response = model([SystemMessage(content=PLAN_PROMPT), HumanMessage(content=topic)])
    plan = response.content
    print("Plan Created:\n", plan)
    return plan

def write_essay(plan: str) -> str:
    response = model([SystemMessage(content=WRITER_PROMPT), HumanMessage(content=plan)])
    essay = response.content
    print("Essay Draft:\n", essay)
    return essay

def critique_essay(essay: str) -> str:
    try:
        truncated_essay = essay[:400]  # Ensure the query length is within the limit
        response = tavily_client.qna_search(query=truncated_essay, search_depth="advanced")
        print("Tavily Response:\n", response)  # Debugging line to inspect the response
        return response  # Directly returning the string response
    except Exception as e:
        print(f"Error during Tavily API call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print("Response Content:", e.response.content)
        return "Error during critique"

def revise_essay(essay: str, critique: str) -> str:
    response = model([SystemMessage(content=REVISION_PROMPT), HumanMessage(content=essay), HumanMessage(content=critique)])
    revised_essay = response.content
    print("Revised Essay:\n", revised_essay)
    return revised_essay


# In[6]:


class StateGraph:
    def __init__(self, initial_state, states, memory):
        self.initial_state = initial_state
        self.states = {state["name"]: state for state in states}
        self.memory = memory
        self.graph = nx.DiGraph()

        for state in states:
            self.graph.add_node(state["name"])
            if "next" in state:
                if callable(state["next"]):
                    next_state = state["next"](initial_state)
                else:
                    next_state = state["next"]
                self.graph.add_edge(state["name"], next_state)

    def get_action(self, state_name):
        state = self.states.get(state_name)
        if state and 'action' in state:
            return state['action']
        return None

    def get_next(self, state_name, current_state):
        state = self.states.get(state_name)
        if state and 'next' in state:
            if callable(state['next']):
                return state['next'](current_state)
            return state['next']
        return None

    def draw_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
        plt.show()

END = "end"


# In[7]:


graph = StateGraph(
    initial_state=AgentState(task="plan", plan="", draft="", critique="", content=[], revision_number=0, max_revisions=3),
    states=[
        {"name": "plan", "next": "write", "action": create_plan},
        {"name": "write", "next": "critique", "action": write_essay},
        {"name": "critique", "next": "revise", "action": critique_essay},
        {"name": "revise", "next": lambda state: "write" if state["revision_number"] < state["max_revisions"] else END, "action": revise_essay},
    ],
    memory=None  # Memory functionality is not implemented in this simplified version
)

# Draw the state graph
graph.draw_graph()


# In[8]:


def run_agent(topic: str):
    state = graph.initial_state
    state['content'].append(topic)
    critique = ""

    while state['task'] != END:
        task = state['task']
        action = graph.get_action(task)
        if not action:
            raise KeyError(f"No action found for task: {task}")

        if task == "revise":
            next_state = action(state['content'][-2], critique)  # Pass essay and critique
        else:
            next_state = action(state['content'][-1])

        state['content'].append(next_state)
        state['task'] = graph.get_next(task, state)

        if state['task'] == "critique":
            critique = next_state  # Store the critique for the revise step

        if state['task'] == "revise":
            state['revision_number'] += 1

    return state['content'][-1]

if __name__ == "__main__":
    topic = "The internal working of a Large Language Model (LLM)"
    final_essay = run_agent(topic)
    print("Final Essay:\n", final_essay)


# In[ ]:




