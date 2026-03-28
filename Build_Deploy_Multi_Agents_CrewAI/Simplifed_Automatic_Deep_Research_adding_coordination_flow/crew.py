import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import EXASearchTool, ScrapeWebsiteTool
from deep_research_crew.guardrails.guardrails import write_report_guardrail # New import
# Removed CustomPlotTool import


@CrewBase
class LowCostResearchCrew:
    """Ultra low-cost research crew (<10k tokens)"""

    # ------------------ AGENTS ------------------

    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="Researcher",
            goal="Find the most important facts about {user_query} using minimal tokens.", # Reverted goal
            backstory="Efficient researcher that only extracts critical information.", # Reverted backstory
            tools=[
                EXASearchTool(
                    api_key=os.getenv("EXA_API_KEY"),
                    max_results=1  # ⚡ minimal
                ),
                ScrapeWebsiteTool()
                # Removed CustomPlotTool
            ],
            llm="gpt-4o-mini",
            verbose=False,
            max_iter=2  # ⚡ very low
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            role="Writer",
            goal="Write a concise, structured report and verify consistency.",
            backstory="Expert in summarizing and validating information efficiently.",
            llm="gpt-4o-mini",
            verbose=False,
            max_iter=2
        )

    # ------------------ TASKS ------------------

    @task
    def research_task(self) -> Task:
        return Task(
            description=(
                "Research '{user_query}'. "
                "Return ONLY the 5 most important facts. " # Reverted description
                "Each fact must be under 20 words and include 1 source."
                # Removed instruction to use plotting tool
            ),
            expected_output=(
                "5 bullet points. Each <=20 words. Each includes 1 source."
                # Removed note about generated plots
            ),
            agent=self.researcher()
        )

    @task
    def write_task(self) -> Task:
        return Task(
            description=(
                "Write a short report based on the research. "
                "Ensure consistency and remove weak claims."
                # Removed plot reference instruction
            ),
            expected_output="""Short report with:
- Title
- 1 paragraph summary (<=100 words)
- 5 key insights
- Sources""", # Reverted expected output
            agent=self.writer(),
            output_callbacks=[write_report_guardrail],
            markdown_output=True,
            output_file='report.md'
        )

    # ------------------ CREW ------------------

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,  # ⚡ critical
            verbose=False,
            tracing=False
        )
