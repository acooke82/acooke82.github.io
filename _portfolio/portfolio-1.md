---
title: "GenAI Agents"
excerpt: "Building and testing agentic workflows for optimal performance. ![Internship Post Image](/images/internship-post-image-1.png)"
collection: portfolio
---
_Tools & Technologies: LangChain, LangSmith, pytest, openai_

### Summary

The main goal of my internship at KSG was to understand, build, and test different GenAI agent workflows to determine which architecture would be best to use in our upcoming product. 

Once the , my team and I were put into a Sprint and assigned tasks to complete for the agent to be developed and deployed. My core task, amongst assisting with other areas in the project, was building a database cache.  

### In-depth Project Overview 

#### 1. Understanding Agents and Agentic Workflows 
##### Supervisor Agent
##### Single Agent 

#### 2. Prompt Engineering
##### LangSmith

### 3. Database Cache

#### Cache

#### Tests & Debugging
Before my code could be deployed and merged to the main dev branch, it needed to be locally tested. I made use of *pytest*, specifically the *MonkeyPatch* fixture, to test my code. I needed to use MonkeyPatch because I did not want to call, or impact, the existing Redis cache. To use MonkeyPatch, I built a "MockCache" class that contained dummy variables to mirror the actual the cache values. I created three different test files, outlined below. 

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

##### test_generate_recipe.py
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


##### test_pull_recipe.py
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
Since I had never used the MonkeyPatch fixture before, it took me some time to correctly set up my testing environment and to understand how to approach mocking the live cache. However, once those issues were addressed, the creation of the tests themselves were straightforward. Once my assertions were all passing, I sent a pull request for my code to be integrated with the project. The database cache was completed at the end of my internship and successfully deployed onto the main product repository. 
