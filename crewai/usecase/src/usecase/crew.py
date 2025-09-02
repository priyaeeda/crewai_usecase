from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# Define the LLM to be used by all agents
llm = LLM(
    model="gemini/gemini-2.0-flash",  # You can replace this with another provider/model
    temperature=0.7,
)

@CrewBase
class TripPlanner():
    """Trip Planner Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # --- Agents ---
    @agent
    def city_selection_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['city_selection_expert'],  # type: ignore[index]
            verbose=True,
            llm=llm
        )

    @agent
    def local_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['local_expert'],  # type: ignore[index]
            verbose=True,
            llm=llm
        )

    @agent
    def travel_concierge(self) -> Agent:
        return Agent(
            config=self.agents_config['travel_concierge'],  # type: ignore[index]
            verbose=True,
            llm=llm
        )

    # --- Tasks ---
    @task
    def city_selection_task(self) -> Task:
        return Task(
            config=self.tasks_config['city_selection_task'],  # type: ignore[index]
        )

    @task
    def local_expert_task(self) -> Task:
        return Task(
            config=self.tasks_config['local_expert_task'],  # type: ignore[index]
        )

    @task
    def travel_concierge_task(self) -> Task:
        return Task(
            config=self.tasks_config['travel_concierge_task'],  # type: ignore[index]
            output_file='trip_plan.md'
        )

    # --- Crew Orchestration ---
    @crew
    def crew(self) -> Crew:
        """Creates the Trip Planner crew"""
        return Crew(
            agents=self.agents,   # Created by @agent decorators
            tasks=self.tasks,     # Created by @task decorators
            process=Process.sequential,  # Run in order: city → local → concierge
            verbose=True,
        )
