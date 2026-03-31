
from crewai import Agent, Crew, Process, Task

class DeepResearchCrew:
    def __init__(self, llm):
        self.llm = llm

    def create_crew(self, query):
        # Placeholder agents and tasks. These would typically be defined
        # in config files or directly here for the actual crew logic.
        researcher = Agent(
            role='Senior Research Analyst',
            goal='Uncover critical insights and data points from the internet.',
            backstory='A meticulous researcher adept at finding hidden information.',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        reporter = Agent(
            role='Report Writer',
            goal='Compile comprehensive and factual reports.',
            backstory='Skilled in synthesizing complex information into clear, concise, and engaging reports.',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        task_research = Task(
            description=f"Conduct a thorough research on '{query}'. Identify key facts, figures, and trends.",
            agent=researcher,
            expected_output='A detailed summary of findings with sources.'
        )

        task_report = Task(
            description='Write a comprehensive report based on the research findings.',
            agent=reporter,
            expected_output='A well-structured report covering all aspects of the research, formatted in markdown.'
        )

        crew = Crew(
            agents=[researcher, reporter],
            tasks=[task_research, task_report],
            process=Process.sequential,
            verbose=True
        )
        return crew

