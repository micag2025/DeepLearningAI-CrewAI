#!/usr/bin/env python
import sys
import os

# Add the 'src' directory to sys.path so 'deep_research_flow' package can be found
# Assuming the script is executed from the 'deep_research_flow' directory (e.g., /content/deep_research_flow)
# and 'src' is a subdirectory within it.
current_dir_when_run = os.getcwd()
src_path = os.path.join(current_dir_when_run, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from random import randint
import json
import time
from typing import List, Dict
from pydantic import BaseModel, Field

from crewai.flow import Flow, listen, start # Corrected import for Flow, listen, start
from crewai import LLM # LLM is correctly imported from crewai
from deep_research_flow.crews.deep_research_crew.deep_research_crew import DeepResearchCrew

# Define our models for structured data
class Section(BaseModel):
    title: str = Field(description="Title of the section")
    description: str = Field(description="Brief description of what the section should cover")

class GuideOutline(BaseModel):
    title: str = Field(description="Title of the guide")
    introduction: str = Field(description="Introduction to the topic")
    target_audience: str = Field(description="Description of the target audience")
    sections: List[Section] = Field(description="List of sections in the guide")
    conclusion: str = Field(description="Conclusion or summary of the guide")

# Define our flow state
class GuideCreatorState(BaseModel):
    topic: str = ""
    audience_level: str = ""
    guide_outline: GuideOutline = None
    sections_content: Dict[str, str] = {}

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Flow for creating a comprehensive guide on any topic"""

    @start()
    def get_user_input(self):
        """Get input from the user about the guide topic and audience"""
        print("\n=== Create Your Comprehensive Guide ===\n")

        # Get user input
        self.state.topic = input("What topic would you like to create a guide for? ")

        # Get audience level with validation
        while True:
            audience = input("Who is your target audience? (beginner/intermediate/advanced) ").lower()
            if audience in ["beginner", "intermediate", "advanced"]:
                self.state.audience_level = audience
                break
            print("Please enter 'beginner', 'intermediate', or 'advanced'")

        print(f"\nCreating a guide on {self.state.topic} for {self.state.audience_level} audience...\n")
        return self.state

    @listen(get_user_input)
    def create_guide_outline(self, state):
        """Create a structured outline for the guide using a direct LLM call"""
        print("Creating guide outline...")

        # Initialize the LLM
        llm = LLM(model="openai/gpt-4o-mini", response_format=GuideOutline)

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"""
            Create a detailed outline for a comprehensive guide on "{state.topic}" for {state.audience_level} level learners.

            The outline should include:
            1. A compelling title for the guide
            2. An introduction to the topic
            3. 4-6 main sections that cover the most important aspects of the topic
            4. A conclusion or summary

            For each section, provide a clear title and a brief description of what it should cover.
            """}
        ]

        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)

        # Parse the JSON response
        outline_dict = response.model_dump()
        self.state.guide_outline = GuideOutline(**outline_dict)

        # Ensure output directory exists before saving
        os.makedirs("output", exist_ok=True)

        # Save the outline to a file
        with open("output/guide_outline.json", "w") as f:
            json.dump(outline_dict, f, indent=2)

        print(f"Guide outline created with {len(self.state.guide_outline.sections)} sections")
        return self.state.guide_outline

    @listen(create_guide_outline)
    def write_and_compile_guide(self, outline):
        """Write all sections and compile the guide"""
        print("Writing guide sections and compiling...")
        llm = LLM(model="openai/gpt-4o-mini") # Re-instantiate LLM for this method if not passed

        deep_research_crew_instance = DeepResearchCrew(llm=llm) # FIX: Pass llm to constructor

        completed_sections = []

        # Process sections one by one to maintain context flow
        for section in outline.sections:
            print(f"Processing section: {section.title}")

            # Build context from previous sections
            previous_sections_text = ""
            if completed_sections:
                previous_sections_text = "# Previously Written Sections\\n\\n"
                for title in completed_sections:
                    # Access the raw string content from CrewOutput objects
                    section_raw_content = self.state.sections_content.get(title, "").raw if hasattr(self.state.sections_content.get(title, ""), 'raw') else self.state.sections_content.get(title, "")
                    previous_sections_text += f"## {title}\\n\\n"
                    previous_sections_text += section_raw_content + "\\n\\n"
            else:
                previous_sections_text = "No previous sections written yet."

            # Construct research_query for DeepResearchCrew
            research_query = f"Conduct thorough research and write a comprehensive section on '{section.title}' for a guide about '{self.state.topic}'. The section should specifically address: '{section.description}'. The target audience is {self.state.audience_level} learners. Incorporate insights from previous sections if relevant: {previous_sections_text}. Ensure the content is well-structured, informative, and engaging."

            # Run the content crew for this section
            crew_result = deep_research_crew_instance.create_crew(query=research_query).kickoff()

            # Store the content (store the CrewOutput object, but extract raw for use)
            self.state.sections_content[section.title] = crew_result
            completed_sections.append(section.title)
            print(f"Section completed: {section.title}")

        # Compile the final guide
        guide_content = f"# {outline.title}\\n\\n"
        guide_content += f"## Introduction\\n\\n{outline.introduction}\\n\\n"

        # Add each section in order
        for section in outline.sections:
            # Access the raw string content from CrewOutput objects here too
            section_content = self.state.sections_content.get(section.title, "").raw if hasattr(self.state.sections_content.get(section.title, ""), 'raw') else self.state.sections_content.get(section.title, "")
            guide_content += f"\\n\\n{section_content}\\n\\n"

        # Add conclusion
        guide_content += f"## Conclusion\\n\\n{outline.conclusion}\\n\\n"

        # Save the guide
        with open("output/complete_guide.md", "w") as f:
            f.write(guide_content)

        print("\\nComplete guide compiled and saved to output/complete_guide.md")
        return "Guide creation completed successfully"

def kickoff():
    """Run the guide creator flow"""
    GuideCreatorFlow().kickoff()
    print("\\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/complete_guide.md to view it.")

def plot():
    """Generate a visualization of the flow"""
    flow = GuideCreatorFlow()
    flow.plot("guide_creator_flow")
    print("Flow visualization saved to guide_creator_flow.html")

if __name__ == "__main__":
    kickoff()
