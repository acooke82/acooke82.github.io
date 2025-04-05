---
title: "GenAI Agent Analysis"
excerpt: "Building and testing agentic workflows for optimal performance. ![Internship Post Image](/images/internship-post-image-1.png)"
collection: portfolio
---
_Tools & Technologies: LangChain, LangSmith, pytest, openai_

### Summary

The main goal of my internship at KSG was to understand, build, and test different GenAI agent workflows to determine which architecture would be best to use in our upcoming product. 

Once the , my team and I were put into a Sprint and assigned tasks to complete for the agent to be developed and deployed. My core task, amongst assisting with other areas in the project, was building a database cache.  

### In-depth Project Overview 

### 1. Understanding Agents and Agentic Workflows 
##### Supervisor Agent
##### Single Agent 

### 2. Prompt Engineering
##### LangSmith

### 3. Database Cache
Once we had defined the type of agent workflow we would be using for the product, my sprint tasks revolved around creating a database cache. The cache needed to handle three use cases: (1) save specific aspects of the agent recipe per agentID, (2) be able to identify and extract the cached information if the given agentID had already been saved, and (3) be able to identify changes to the recipe on the frontend and update the values on the database automatically.

#### Building the Cache
There were a few ways to approach the caching problem. Handling the recipe caching itself, and fetching the information, was relatively straightforward. However, updating the database values automatically proved to be more challenging. A simple fix regarding the updates could be to set the time to live of the cache (TTL) as very short, forcing it to regularly be regenerated. However, setting a short TTL would increase server load and effectively remove the need for a cache. My first written attempt was to give pull_recipe a boolean parameter that would be "True" if the agent was out of date or "False" otherwise. If given "True", generate_recipe would be prompted to run to re-generate the agent recipe. However, upon discussion with more senior team members, we determined that it would be more appropriate to push updates to the cache through the crud.py file, as that file already contained methods to get, validate, and update agent resources. Please see below for the main cache functions as well as an example of the cache features integrated with the crud.py file. Please note for the below code, "..." indicates an area where another team member has written code. I have isolated these snippets to just my work.

```python

from typing import Union
from uuid import UUID

from fastapi import Depends

from initialization import cache, logger
from initialization.dbConnection import dbConnector
from middlewares.agent import validate_agent

from ...custom_api.models import CustomAPI
from ...integrations.models import ConnectedTool, IntegrationProvider
from ...knowledge_bases import KnowledgeBase, KnowledgeBaseCategory
from ..models import Agent
from .models import (
    ApiChannelArgs,
    ComposioToolRecipe,
    CustomApiToolRecipe,
    DatabaseKnowledgeBaseToolRecipe,
    Recipe,
    WebsiteKnowledgeBaseToolRecipe,
)

...
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
...
await db.flush()
    if commit:
        await db.commit()
    await db.refresh(agentInDb)

    await generate_recipe(agentInDb.id, db)
    return agentInDb
...

```
#### Tests & Debugging
Before my code could be deployed and merged to the main dev branch, it needed to be locally tested. I made use of *pytest*, specifically the *MonkeyPatch* fixture, to test my code. I needed to use MonkeyPatch because I did not want to call, or impact, the actual Redis cache. To use MonkeyPatch, I built a "MockCache" class that contained dummy variables to mirror the actual the cache values. I created three different test files, outlined below. 

 ##### test_recipe_builder.py
 The goal of my first testing file was to confirm that the recipe_builder function was working as expected and so that I could confirm the output of recipe_builder. I needed to verify this in order to accurately 
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
