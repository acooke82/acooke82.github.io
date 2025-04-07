---
title: "GenAI Agent Analysis"
excerpt: "Building and testing agentic workflows for optimal performance. ![Internship Post Image](/images/internship-post-image-1.png)"
collection: portfolio
---
_Tools & Technologies: LangChain, LangGraph, LangSmith, Pytest_

### Summary

The main goal of my internship at KSG was to understand, build, and test different GenAI agent workflows to determine which architecture would be best to use in our upcoming product. Once the architecture was determined, my team and I would be assigned tasks to complete developing and deploying the rest of the product. As the project progressed, my core tasks, amongst assisting with other areas in the project, were building a database cache and creating prompts that would effectively make use of our new tools and architecture. In addition to completing work for the company, my personal goal was to become more comfortable coding in a professional environment. I also wanted to build a stronger foundation of my understanding and practical experience with large language models.   

### 1. Understanding Agents and Agentic Workflows 

An *agent* is a system that uses a large language model (LLM) to perform tasks on behalf of a user or within a larger system. They have access to tools and data that help them make decisions. Testing different types of workflows is important because each system comes with its own pros and cons and you want to ensure you are building the best system for your product as it may be impossible or difficult to alter later. To build our agent workflows, we used the libraries *LangGraph* and *LangChain*. LangGraph enables developers to build complex agent applications quickly as it has built-in graph structure, state management, and coordination support. LangGraph works by using *nodes*, which are functions that perform specific tasks such as calling tools, *edges*, which control the flow of information between nodes, and *states*, which are objects in the graph such as conversation history or internal variables. LangChain, on the other hand, focuses on simpler linear builds, where you add each step to the established chain to execute tasks in a specific order. 

Because I did not have experience with either of these libraries, or with agents, it took me some time to do research and analysis. With LangChain, you need to understand how to build a ReAct agent, agent tools, a prompt, and how to configure an LLM to set up the chain. Ideally, you want to understand LangChain before working in LangGraph, as LangGraph requires the same steps but with multiple agents and prompts, complicating how to build the chain. You will be able to see some of these differences in working with these libraries in the code for each workflow below. My supervisor had built an example multi-agent workflow in steps using Jupyter Notebooks that I could follow along with to understand each piece better, but working with LangGraph first meant that I had a much steeper learning curve. LangChain has "LangChain Academy" with courses on LangGraph and plenty of documentation online which was very beneficial in navigating these libraries. With these tools at my disposal, I was able to begin working on the workflows.  

##### Multi-Agent Workflow

The first workflow that I tested involved a *supervisor agent*. A supervisor agent is part of a multi-agent workflow in which one agent, the *supervisor*, serves as the controller of the other agents and handles communication with the user. The supervisor agent itself does not have any tools and must make the decision on which agents under it to call in order to properly complete the task. Each agent under the supervisor has their own prompt, purpose, and tools, and cannot interact directly with the user. In this workflow, agents are typically their own graph node. They are routed a task, after which they can decide to end the execution or send their response to another agent. Please see below for insight into the multi-agent workflow I was testing, built using the example from my supervisor:

```python
def build_async_workflow(csv_file_path: str ="all-states-history.csv", 
                         api_file_path: str ="openapi_kraken.json"):
    """
    Creates the LLM, specialized agents, and the async StateGraph
    that orchestrates them with a supervisor node. Returns the
    not-yet-compiled workflow. You can then compile it with a checkpointer.
    """
    logger.debug("build_async_workflow: Starting building workflow")

    # ------------------------------------
    # A) Create LLM
    # ------------------------------------
    model_name = os.environ.get("GPT4o_DEPLOYMENT_NAME", "")
    logger.debug("Creating LLM with deployment_name=%s", model_name)

    llm = AzureChatOpenAI(
        deployment_name=model_name,
        temperature=0,
        max_tokens=2000,
        streaming=True,  # set True if you want partial streaming from the LLM
    )

    # ------------------------------------
    # B) Create specialized agents
    # ------------------------------------
    logger.debug("Creating docsearch_agent, csvsearch_agent, sqlsearch_agent, websearch_agent, apisearch_agent")

    docsearch_agent = create_docsearch_agent(
        llm=llm,
        indexes=["srch-index-files", "srch-index-csv", "srch-index-books"],
        k=20,
        reranker_th=1.5,
        prompt=CUSTOM_CHATBOT_PREFIX + DOCSEARCH_PROMPT_TEXT,
        sas_token=os.environ.get("BLOB_SAS_TOKEN", "")
    )

    csvsearch_agent = create_csvsearch_agent(
        llm=llm,
        prompt=CUSTOM_CHATBOT_PREFIX + CSV_AGENT_PROMPT_TEXT.format(
            file_url=str(csv_file_path)
        )
    )

    sqlsearch_agent = create_sqlsearch_agent(
        llm=llm,
        prompt=CUSTOM_CHATBOT_PREFIX + MSSQL_AGENT_PROMPT_TEXT
    )

    websearch_agent = create_websearch_agent(
        llm=llm,
        prompt=CUSTOM_CHATBOT_PREFIX + BING_PROMPT_TEXT
    )

    logger.debug("Reading API openapi_kraken.json from %s", api_file_path)
    with open(api_file_path, "r") as file:
        spec = json.load(file)
    reduced_api_spec = reduce_openapi_spec(spec)

    apisearch_agent = create_apisearch_agent(
        llm=llm,
        prompt=CUSTOM_CHATBOT_PREFIX + APISEARCH_PROMPT_TEXT.format(
            api_spec=reduced_api_spec
        )
    )
    # ------------------------------------
    # C) Build the async LangGraph
    # ------------------------------------
    logger.debug("Building the StateGraph for multi-agent workflow")
    workflow = StateGraph(AgentState)

    sup_node = functools.partial(supervisor_node_async, llm=llm)
    workflow.add_node("supervisor", sup_node)

    doc_node = functools.partial(agent_node_async, agent=docsearch_agent, name="DocSearchAgent")
    csv_node = functools.partial(agent_node_async, agent=csvsearch_agent, name="CSVSearchAgent")
    sql_node = functools.partial(agent_node_async, agent=sqlsearch_agent, name="SQLSearchAgent")
    web_node = functools.partial(agent_node_async, agent=websearch_agent, name="WebSearchAgent")
    api_node = functools.partial(agent_node_async, agent=apisearch_agent, name="APISearchAgent")

    workflow.add_node("DocSearchAgent", doc_node)
    workflow.add_node("CSVSearchAgent", csv_node)
    workflow.add_node("SQLSearchAgent", sql_node)
    workflow.add_node("WebSearchAgent", web_node)
    workflow.add_node("APISearchAgent", api_node)

    for agent_name in ["DocSearchAgent", "CSVSearchAgent", "SQLSearchAgent", "WebSearchAgent", "APISearchAgent"]:
        workflow.add_edge(agent_name, "supervisor")

    conditional_map = {
        "DocSearchAgent": "DocSearchAgent",
        "SQLSearchAgent": "SQLSearchAgent",
        "CSVSearchAgent": "CSVSearchAgent",
        "WebSearchAgent": "WebSearchAgent",
        "APISearchAgent": "APISearchAgent",
        "FINISH": END
    }
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.add_edge(START, "supervisor")

    logger.debug("build_async_workflow: Workflow build complete")
    return workflow
```

A multi-agent workflow has the advantages of having separate agents for each task, making it easy to add and maintain agents without disrupting the original workflow. It also allows you to have more specialization, as each agent can become an expert of a specific task or domain. This leads to a personalized system with overall good performance. On the other hand, a multi-agent workflow is expensive to maintain, complex to design, and requires careful agent coordination. This led us to explore different options. 

##### Single Agent Workflow 

The second workflow that I tested involved a *single agent*. In a single agent workflow, one agent interacts directly with the user and has all the tools. It must determine which tool to use or which order to use them in by itself to resolve the user's query. I was able to tweak the multi-agent workflow to rework it as a single agent that had similar tools and objectives, shown below: 

```python
# 1) Define Tools (Expert Functions)
# -----------------------------------------------------------------------------
COMPLETION_TOKENS = 1500
llm = AzureChatOpenAI(
    deployment_name=os.environ.get("GPT4o_DEPLOYMENT_NAME", ""),
    temperature=0,
    max_tokens=COMPLETION_TOKENS,
    streaming=True,
    api_version="2024-10-01-preview"
)

requests_wrapper = TextRequestsWrapper()
toolkit = RequestsToolkit(requests_wrapper=requests_wrapper, allow_dangerous_requests=True)

db_config = {
    'drivername': 'mssql+pyodbc',
    'username': os.environ["SQL_SERVER_USERNAME"] + '@' + os.environ["SQL_SERVER_NAME"],
    'password': os.environ["SQL_SERVER_PASSWORD"],
    'host': os.environ["SQL_SERVER_NAME"],
    'port': 1433,
    'database': os.environ["SQL_SERVER_DATABASE"],
    'query': {'driver': 'ODBC Driver 17 for SQL Server'},
}
db_url = URL.create(**db_config)
sqltoolkit = SQLDatabaseToolkit(db=SQLDatabase.from_uri(db_url), llm=llm)

TOOLS = [
    GetBingSearchResults_Tool(
        name="WebSearcher",
        description="useful to find information about a product or issue on the web.\n",
    ),
    FetchWebPageTool(
        name="WebPageFetcher",
        description="Useful for fetching the content, image URLs, and links of a web page/URL/link.\n",
        max_words=10000, 
        images=True, 
        links=True
    ),
    DocumentSearcher(
        name="DocumentSearcher", 
        description="Useful for searching for documents on the web.\n",
        indexes=["srch-index-files", "srch-index-csv", "srch-index-books"],
        k=10,
        reranker_th=1,
        sas_token=os.environ["BLOB_SAS_TOKEN"]
    ),
    PythonAstREPLTool()
]
TOOLS = TOOLS + sqltoolkit.get_tools() + toolkit.get_tools()

# -----------------------------------------------------------------------------
# 2) Initialize the LLM with Tool Support
# -----------------------------------------------------------------------------

llm_with_tools = llm.bind_tools(TOOLS)

# -----------------------------------------------------------------------------
# 3) Cache Prompt and Setup Trimmer and Chain
# -----------------------------------------------------------------------------
PROMPT = hub.pull("acooke/single_agent_prompt_text:becb1153")
if not PROMPT:
    raise ValueError("Failed to fetch prompt from hub or prompt was empty.")

TRIMMER = trim_messages(
    max_tokens=30,
    strategy="last",
    token_counter=len,
    include_system=True
)

CHAIN = PROMPT | TRIMMER | llm_with_tools

# -----------------------------------------------------------------------------
# 4) Define the Chatbot Node
# -----------------------------------------------------------------------------


async def call_model(state: MessagesState, config: RunnableConfig):
    """
    Handles conversation logic, constructs prompts, trims messages,
    and interacts with the LLM.
    """
    messages = state["messages"]
    
    try:
        response = await CHAIN.ainvoke(state["messages"], config)
    except Exception as e:
        logger.error(f"Error during LLM invocation: {e}")

    # Filter to only keep relevant message types
    filtered_messages = filter_messages(
        messages + [response],
        include_types=[SystemMessage, HumanMessage, AIMessage]
    )

    return {"messages": filtered_messages}


def should_continue(state: MessagesState):
    """
    Determines whether to proceed to tool execution or end the workflow.
    """

    messages = state["messages"]
    last_message = messages[-1]
    
    # If the AI indicates a tool call, route to 'tools', else end.
    return "tools" if hasattr(last_message, 'tool_calls') and last_message.tool_calls else END

# -----------------------------------------------------------------------------
# 5) Build the Async Workflow
# -----------------------------------------------------------------------------


def build_async_workflow(csv_file_path, api_file_path) -> StateGraph:
    """
    Constructs and returns a LangGraph-based workflow:
      - Starts with 'agent' node (LLM interaction).
      - Routes to 'tools' node if tool calls are detected.
      - Returns to 'agent' node after tool execution.
    """
    logger.debug("Starting build of the async RAG bot workflow.")

    workflow = StateGraph(MessagesState)

    # Define ToolNode
    tool_node = ToolNode(tools=TOOLS)

    # Define nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Define edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    logger.debug("Async Basic RAG bot workflow build is complete.")

    return workflow
```

Comparing it with the multi-agent workflow above, we can see a few key differences. A single agent contains less nodes, the tools are directly passed to it, and there is only one prompt. A single agent workflow is easy to implement, cost-effective, and quick to develop. However, because one agent contains all of the software, it lacks scalability, specialization, and fault tolerance. Ultimately, after discussing with other developers we decided to go with an "agents as tools" workflow. This is an extension of the single agent system, in which you can build one or many agents with their own expertise and link them together if needed. There is no supervisor agent in this workflow and each agent has a specialty.  

### 2. Prompt Engineering

Once we had determined our architecture and the system was underway, we needed to build and test prompts, a process called *prompt engineering*. Prompt engineering is how we can ensure that we are effectively using our agents to produce the most accurate and relevant outputs. Because we were building single agents, it was extremely important that we got the prompts correct, as a single agent needs to know when exactly to use each tool and for what purpose to use it. I had not done any prompt engineering before, but I enjoyed it because you can see how your changes impact the output immediately and there are many tools or platforms to help you do so. In regards to our product, it was also fun to build prompts for specific customers because they all have specific needs for tone and objectives. This means that each prompt is unique and you can get pretty creative! It's always a bit fun to see what the agent produces, as sometimes the results are not what you were expecting. Prompts for single agents tend to be very long, as they need to outline the general capabilities of the agent, safety and privacy rules, how to interact with the user, source material, tool definitions and tool instructions, example outputs, and company information (if needed). For example, the prompts I were building were at least four pages long and could stretch to seven. 

##### LangSmith

The quickest way to test your prompts before making them live is to host them on a platform intended for testing, managing, and evaluating LLMs. Since we were already using LangChain, it was simplest to work with *LangSmith*. LangSmith gives you the ability to build prompts, specify their input/output parameters, and test them all within the UI before setting them in your code. Once you're done editing, you can implement them into your chain like so:   

```python
PROMPT = hub.pull("acooke/single_agent_prompt_text:becb1153")
if not PROMPT:
    raise ValueError("Failed to fetch prompt from hub or prompt was empty.")

TRIMMER = trim_messages(
    max_tokens=30,
    strategy="last",
    token_counter=len,
    include_system=True
)

CHAIN = PROMPT | TRIMMER | llm_with_tools
```

### 3. Database Cache
With our agent architecture complete and prompts in action, my sprint tasks revolved around creating a database cache. The cache needed to handle three use cases: (1) save specific aspects of the agent recipe per agentID, (2) be able to identify and extract the cached information if the given agentID had already been saved, and (3) be able to identify changes to the recipe on the frontend and update the values on the database automatically.

#### Building the Cache
There were a few ways to approach the caching problem. Handling the recipe caching itself and fetching the information was relatively straightforward. However, updating the database values automatically proved to be more challenging. A simple fix regarding the updates could be to set the time to live of the cache (TTL) as very short, forcing it to regularly be regenerated. However, setting a short TTL would increase server load and effectively remove the need for a cache. My first attempt was to give pull_recipe a boolean parameter that would be "True" if the agent was out of date or "False" otherwise. If given "True", generate_recipe would be prompted to run to re-generate the agent recipe. However, upon discussion with more senior team members, we determined that it would be more appropriate to push updates to the cache through the crud.py file, as that file already contained methods to get, validate, and update agent resources. Please see below for the main cache functions as well as an example of the cache features integrated with the crud.py file:

```python
async def pull_recipe(agentId: UUID, db=Depends(dbConnector.get_db)) -> Recipe:
    # recipe = await cache.get(str(agentId))
    recipe = None
    if recipe:
        logger.log_debug(f"Recipe found in cache for agent {agentId}!")
        return Recipe.model_validate_json(recipe)

    logger.log_debug(f"No recipe found in cache for agent {agentId}. Creating a new one!")
    recipe = await generate_recipe(agentId, db)
    return recipe


async def generate_recipe(agentId: UUID, db=Depends(dbConnector.get_db)) -> Recipe:
    recipe = await recipe_builder(agentId, db)
    recipeJson = recipe.model_dump_json()
    await cache.set(key=str(agentId), value=recipeJson, ttl=432000)
    return recipe
```

```python
await db.flush()
    if commit:
        await db.commit()
    await db.refresh(agentInDb)

    await generate_recipe(agentInDb.id, db)
    return agentInDb
```

#### Tests & Debugging
Before my code could be deployed and merged to the main dev branch, it needed to be locally tested. I made use of *Pytest*, specifically the *MonkeyPatch* fixture, to test my code. I needed to use MonkeyPatch because I did not want to call, or impact, the actual Redis cache. To use MonkeyPatch, I built a "MockCache" class that contained dummy variables to mirror the actual cache values. I created three different test files, outlined below. 

##### test_recipe_builder.py
The goal of my first testing file was to confirm that the recipe_builder function was working as expected and so that I could 
confirm the output of recipe_builder. I needed to verify this in order to accurately 
build the generate_recipe and pull_recipe functions that would be using that output.

```python
from sqlalchemy.ext.asyncio import AsyncSession

from services.agents.models import Agent
from services.agents.recipes.models import Recipe
from services.agents.recipes.utils import recipe_builder


async def test_recipe_builder_successful(agent_1: Agent, db_session: AsyncSession):
    agentId = agent_1.id
    recipe = await recipe_builder(agentId, db=db_session)

    assert isinstance(recipe, Recipe)
    assert recipe.name == "Test Agent 1"
```

#### test_generate_recipe.py
The goal of my second testing file was to confirm that generate_recipe would correctly identify the agentId to build and cache the recipe for that unique Id. I asserted that the generated recipe was as expected by testing it against an example agent that we knew the values for. 

```python
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import AsyncSession

from services.agents.models import Agent
from services.agents.recipes.models import Recipe
from services.agents.recipes.utils import generate_recipe


class MockCache:
    async def get(self, key, value, ttl):
        return None

    async def set(self, key, value, ttl):
        return None


async def test_generate_recipe_successful(
    agent_1: Agent, db_session: AsyncSession, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr("services.agents.recipes.utils.cache", MockCache())
    agentId = agent_1.id
    recipe = await generate_recipe(agentId, db=db_session)

    assert isinstance(recipe, Recipe)
    assert recipe.name == agent_1.name
    assert recipe.description == agent_1.description
    assert recipe.prompt == agent_1.promptVersion.content
```


#### test_pull_recipe.py
The goal of my third testing file was to confirm that the pull_recipe function would be able to successfully complete two actions. One, that it could pull a recipe from the cache if it existed. Two, that if the recipe requested did not yet exist, it would trigger an automatic generation of a new recipe.

```python
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import AsyncSession

from services.agents.models import Agent
from services.agents.recipes.models import Recipe
from services.agents.recipes.utils import pull_recipe


class MockCache:
    def __init__(self):
        self.cache = {}

    async def get(self, key):
        return self.cache.get(key)

    async def set(self, key, value, ttl=None):
        self.cache[key] = value

    def clear(self):
        self.cache = {}


# Test that the recipe is pulled from the cache if it exists
async def test_pull_recipe_successful(
    agent_1: Agent, db_session: AsyncSession, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr("services.agents.recipes.utils.cache", MockCache())
    agentId = agent_1.id
    recipe = await pull_recipe(agentId, db=db_session)

    assert isinstance(recipe, Recipe)
    assert recipe.name == agent_1.name
    assert recipe.description == agent_1.description


# Test that a new recipe is generated if it does not exist in the cache
async def test_pull_recipe_with_generation_successful(
    agent_1: Agent, db_session: AsyncSession, monkeypatch: MonkeyPatch
):
    mockCache = MockCache()
    monkeypatch.setattr("services.agents.recipes.utils.cache", mockCache)
    agentId = agent_1.id

    recipe = await pull_recipe(agentId, db=db_session)

    assert str(agentId) in mockCache.cache
    assert isinstance(recipe, Recipe)
```
As I had never used the MonkeyPatch fixture before, it took me some time to correctly set up my testing environment and to understand how to approach mocking the live cache. However, once those issues were addressed, the creation of the tests themselves were straightforward. Once my assertions were all passing, I sent a pull request for my tests and cache code to be integrated with the project. The database cache was completed at the end of my internship and successfully integrated into the product. 
