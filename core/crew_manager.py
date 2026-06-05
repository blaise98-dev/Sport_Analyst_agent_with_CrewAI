"""
Basketball Analytics CrewAI Manager

This module provides a clean, optimized crew manager for basketball analytics
using CrewAI framework. It supports both dynamic YAML-based configuration 
and structured class-based crew definitions.
"""

import os
from typing import Dict, Any
import textwrap

from crewai import Agent, Task, Crew, LLM, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    ScrapeWebsiteTool,
    SerperDevTool,
    # Add other tools as needed
)


class CrewManager:
    """
    Dynamic crew builder that creates agents and tasks from YAML configurations.
    Used for flexible, configuration-driven crew creation.
    """

    @staticmethod
    def create_agent(name: str, config: Dict[str, Any], llm) -> Agent:
        """
        Create a CrewAI agent from configuration dictionary.
        
        Args:
            name: Agent identifier
            config: Agent configuration with role, goal, backstory
            llm: Language model instance
            
        Returns:
            Configured Agent instance
        """
        # Validate required configuration
        role = config.get("role", "").strip()
        goal = config.get("goal", "").strip()
        backstory = config.get("backstory", "").strip()
        
        if not role or not goal:
            raise ValueError(f"Agent '{name}' requires 'role' and 'goal' fields")

        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=llm,
            allow_delegation=config.get("allow_delegation", False),
            verbose=config.get("verbose", True),
            allow_code_execution=False  # Security: Disable code execution
        )

    @staticmethod
    def create_task(name: str, config: Dict[str, Any], agent: Agent, data_context: str = "") -> Task:
        """
        Create a CrewAI task from configuration dictionary.
        
        Args:
            name: Task identifier
            config: Task configuration with description and expected_output
            agent: Agent assigned to this task
            data_context: Additional context data (e.g., CSV sample)
            
        Returns:
            Configured Task instance
        """
        description = config.get("description", "").strip()
        expected_output = config.get("expected_output", "").strip()

        if not description or not expected_output:
            raise ValueError(f"Task '{name}' requires 'description' and 'expected_output' fields")

        # Enhance description with data context if provided
        if data_context:
            enhanced_description = textwrap.dedent(f"""
            {description}

            Data Context (Sample):
            ```csv
            {data_context}
            ```
            """).strip()
        else:
            enhanced_description = description

        return Task(
            description=enhanced_description,
            expected_output=expected_output,
            agent=agent
        )

    @staticmethod
    def build_crew(agents_config: Dict[str, Any], tasks_config: Dict[str, Any], 
                   llm, dataframe=None) -> Crew:
        """
        Build a complete crew from YAML configurations.
        
        Args:
            agents_config: Dictionary of agent configurations
            tasks_config: Dictionary of task configurations
            llm: Language model for all agents
            dataframe: Optional data context for tasks
            
        Returns:
            Configured Crew instance ready for execution
        """
        # Validate inputs
        if not agents_config:
            raise ValueError("No agent configurations provided")
        if not tasks_config:
            raise ValueError("No task configurations provided")

        # Create all agents
        agents = {}
        for agent_name, agent_config in agents_config.items():
            agents[agent_name] = CrewManager.create_agent(agent_name, agent_config, llm)

        # Prepare data context for tasks
        data_context = ""
        if dataframe is not None:
            csv_content = dataframe.to_csv(index=False)
            data_context = csv_content[:2000]  # Limit context size

        # Create all tasks with agent assignments
        tasks = {}
        for task_name, task_config in tasks_config.items():
            agent_name = task_config.get("agent")
            if not agent_name:
                raise ValueError(f"Task '{task_name}' missing 'agent' assignment")
            if agent_name not in agents:
                available_agents = list(agents.keys())
                raise ValueError(f"Task '{task_name}' references unknown agent '{agent_name}'. Available: {available_agents}")

            assigned_agent = agents[agent_name]
            tasks[task_name] = CrewManager.create_task(task_name, task_config, assigned_agent, data_context)

        # Build and return crew
        return Crew(
            agents=list(agents.values()),
            tasks=list(tasks.values()),
            verbose=True,
            process=Process.sequential
        )


@CrewBase
class BasketballAnalyticsCrewOptimized:
    """
    Optimized basketball analytics crew with structured agent definitions.
    Each agent has specific tools and responsibilities for basketball data analysis.
    """

    def _create_llm(self, model: str = "ollama/llama3.1:8b", temperature: float = 0.2) -> LLM:
        """Create LLM instance with consistent configuration"""
        return LLM(model=model, temperature=temperature)

    # ===== AGENTS =====
    
    @agent
    def data_collector(self) -> Agent:
        """
        Specialized agent for collecting basketball data from various sources.
        Equipped with web scraping and search tools.
        """
        return Agent(
            config=self.agents_config["basketball_data_collector"],
            tools=[
                ScrapeWebsiteTool(),  # For direct website data extraction
                SerperDevTool()       # For intelligent web searches
            ],
            llm=self._create_llm(temperature=0.1),  # Low temperature for factual data
            allow_delegation=False,
            max_iter=25,
            verbose=True
        )

    @agent
    def operations_consultant(self) -> Agent:
        """
        Strategic analyst focused on basketball operations and tactical recommendations.
        No external tools needed - relies on analysis and reasoning.
        """
        return Agent(
            config=self.agents_config["basketball_operations_consultant"],
            tools=[],  # Pure analytical agent
            llm=self._create_llm(temperature=0.3),  # Moderate creativity for insights
            allow_delegation=False,
            max_iter=20,
            verbose=True
        )

    @agent
    def intelligence_analyst(self) -> Agent:
        """
        Deep analysis specialist for pattern recognition and performance insights.
        Focuses on statistical analysis and trend identification.
        """
        return Agent(
            config=self.agents_config["basketball_intelligence_analyst"],
            tools=[],  # Analytical processing only
            llm=self._create_llm(temperature=0.2),  # Balanced for analytical work
            allow_delegation=False,
            max_iter=20,
            verbose=True
        )

    @agent
    def visualization_expert(self) -> Agent:
        """
        Specialist in data visualization recommendations and chart design.
        Focuses on creating actionable visual insights.
        """
        return Agent(
            config=self.agents_config["basketball_data_visualization_expert"],
            tools=[],  # Visualization planning only
            llm=self._create_llm(temperature=0.4),  # Higher creativity for visual design
            allow_delegation=False,
            max_iter=15,
            verbose=True
        )

    # ===== TASKS =====

    @task
    def collect_basketball_data(self) -> Task:
        """Data collection task - gather comprehensive basketball information"""
        return Task(
            config=self.tasks_config["collect_basketball_data"],
            agent=self.data_collector(),
            output_file="collected_data.json"  # Save collected data
        )

    @task
    def analyze_performance(self) -> Task:
        """Performance analysis task - identify patterns and insights"""
        return Task(
            config=self.tasks_config["generate_performance_analysis"],
            agent=self.operations_consultant(),
            context=[self.collect_basketball_data()]  # Depends on data collection
        )

    @task
    def create_statistics_summary(self) -> Task:
        """Statistical summary task - create comprehensive metrics tables"""
        return Task(
            config=self.tasks_config["create_statistical_summary"],
            agent=self.intelligence_analyst(),
            context=[self.collect_basketball_data()]  # Depends on data collection
        )

    @task
    def design_visualizations(self) -> Task:
        """Visualization design task - recommend charts and visual insights"""
        return Task(
            config=self.tasks_config["design_visualization_recommendations"],
            agent=self.visualization_expert(),
            context=[
                self.analyze_performance(),
                self.create_statistics_summary()
            ]  # Depends on analysis results
        )

    @task
    def compile_final_report(self) -> Task:
        """Final compilation task - create comprehensive basketball intelligence report"""
        return Task(
            config=self.tasks_config["compile_basketball_intelligence_report"],
            agent=self.intelligence_analyst(),
            context=[
                self.collect_basketball_data(),
                self.analyze_performance(),
                self.create_statistics_summary(),
                self.design_visualizations()
            ],  # Aggregates all previous work
            output_file="basketball_intelligence_report.md"
        )

    # ===== CREW ASSEMBLY =====

    @crew
    def crew(self) -> Crew:
        """
        Assemble the complete basketball analytics crew with proper task dependencies.
        Tasks execute sequentially with clear data flow between agents.
        """
        return Crew(
            agents=self.agents,  # Auto-populated by @agent decorator
            tasks=self.tasks,    # Auto-populated by @task decorator
            process=Process.sequential,  # Execute tasks in dependency order
            verbose=True,
            memory=True,  # Enable crew memory for better context sharing
            planning=True  # Enable planning for better task coordination
        )


# ===== UTILITY FUNCTIONS =====

def create_basketball_crew(backend: str, api_key: str, model_name: str, 
                          agents_config: Dict, tasks_config: Dict, dataframe=None) -> Crew:
    """
    Factory function to create basketball analytics crew with specified LLM backend.
    
    Args:
        backend: LLM backend type ('ollama', 'openai', 'gemini')
        api_key: API key for the backend
        model_name: Model name to use
        agents_config: Agent configurations from YAML
        tasks_config: Task configurations from YAML
        dataframe: Optional basketball data for context
        
    Returns:
        Configured Crew ready for basketball analytics
    """
    # Configure LLM based on backend
    if backend.lower() == "ollama":
        llm = LLM(
            model=model_name,
            base_url="http://localhost:11434",
            api_key=api_key or "ollama"
        )
    elif backend.lower() == "openai":
        llm = LLM(
            model=model_name,
            api_key=api_key
        )
    elif backend.lower() == "gemini":
        llm = LLM(
            model=model_name,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Build and return crew
    return CrewManager.build_crew(agents_config, tasks_config, llm, dataframe)


def validate_crew_config(agents_config: Dict, tasks_config: Dict) -> bool:
    """
    Validate crew configuration for completeness and consistency.
    
    Args:
        agents_config: Agent configurations
        tasks_config: Task configurations
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check agents
    required_agent_fields = ["role", "goal", "backstory"]
    for agent_name, config in agents_config.items():
        for field in required_agent_fields:
            if not config.get(field):
                raise ValueError(f"Agent '{agent_name}' missing required field: {field}")

    # Check tasks
    required_task_fields = ["description", "expected_output", "agent"]
    for task_name, config in tasks_config.items():
        for field in required_task_fields:
            if not config.get(field):
                raise ValueError(f"Task '{task_name}' missing required field: {field}")
        
        # Verify agent assignment
        agent_name = config["agent"]
        if agent_name not in agents_config:
            raise ValueError(f"Task '{task_name}' assigned to unknown agent: {agent_name}")

    return True