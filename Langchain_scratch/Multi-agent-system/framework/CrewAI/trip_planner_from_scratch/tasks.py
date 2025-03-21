# To know more about the Task class, visit: https://docs.crewai.com/concepts/tasks
from crewai import Task
from textwrap import dedent

"""
Creating Tasks Cheat Sheet:
👉Begin with the end in mind. Identify the specific outcome your tasks are aiming to achieve.
- Break down the outcome into actionable tasks, assigning each task to the appropriate agent.
- Ensure tasks are descriptive, providing clear instructions and expected deliverables.

Goal:
- Develop a detailed itinerary, including city selection, attractions, and practical travel advice.

Key Steps for Task Creation:
1. Identify the Desired Outcome: Define what success looks like for your project.
    - A detailed itinerary, including city selection, attractions, and practical travel advice.

2. Task Breakdown: Divide the goal into smaller, manageable tasks that agents can execute.
    - Itinerary Planning: develop a detailed plan for each day of the trip.
    - City Selection: Analyze and pick the best cities to visit.
    - Local Tour Guide: Find a local expert to provide insights and recommendations.


3. Assign Tasks to Agents: Match tasks with agents based on their roles and expertise.
    - Itinerary Planning -> Expert Travel Agent
    - City Selection -> City Selection Expert
    - Local Tour Guide -> Local Tour Guide

4. Task Description Template:
   - Use this template as a guide to define each task in your CrewAI application.
   - This template helps ensure that each task is clearly defined, actionable, and aligned with the specific
     goals of your project.

Template:
----------
def [task_name](self, agent, [parameters]):
    return Task(description=dedent(f'''
    **Task**: [Provide a concise name or summary of the task.]
    **Description**: [Detailed description of what the agent is expected to do, including actionable steps and expected outcomes.]
    
    **Parameters**:
    - [Parameter 1]: [Description]
    - [Parameter 2]: [Description]
    ... [Add more parameters as needed.]
    
    **Note**: [Optional section for incentives or encouragement for high-quality work. This can include tips, additional information, etc.]
    '''), agent=agent)
"""


class TripPlannerTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"
    
    def select_destination(self, agent, destination_type, budget, duration):
        return Task(
            description=dedent(
                f"""
                **Task**: Analyze and select the best travel destinations
                **Description**: Based on the traveler's preferences, budget constraints, and time frame, analyze and recommend the 
                optimal travel destinations. Your analysis should consider factors such as seasonal conditions, tourist attractions, 
                safety, and how well the destination matches the traveler's interests.
                
                Please provide:
                1. A ranked list of 3-5 recommended destinations that match the criteria
                2. For each destination, include:
                   - Brief overview and why it's suitable for this trip
                   - Best areas/neighborhoods to stay
                   - Must-see attractions and activities
                   - Estimated costs for accommodations, food, and activities
                   - Best time to visit and current seasonal conditions
                
                **Parameters**:
                - Destination Type: {destination_type} (e.g., beach, city, mountains, cultural)
                - Budget: ${budget} for the entire trip
                - Duration: {duration} days
                
                {self.__tip_section()}
                
                Make sure to use the most recent data possible and consider current travel conditions.
                """
            ),
            expected_output="A detailed report of recommended destinations with justifications and practical information",
            agent=agent,
        )
    
    def create_itinerary(self, agent, destination, duration, interests):
        return Task(
            description=dedent(
                f"""
                **Task**: Develop a comprehensive day-by-day travel itinerary
                **Description**: Create a detailed itinerary for a {duration}-day trip to {destination}, taking into account 
                the traveler's interests and preferences. The itinerary should maximize the travel experience while remaining 
                practical and avoiding an overly rushed schedule.
                
                Your itinerary should include:
                1. A day-by-day schedule with morning, afternoon, and evening activities
                2. Recommended accommodations for each night
                3. Transportation options between sites and activities
                4. Estimated timing for each activity, including travel time between locations
                5. Meal recommendations (especially local specialties worth trying)
                6. Backup activities or rainy-day alternatives
                7. A packing list tailored to the destination and planned activities
                8. Estimated budget breakdown for each day
                
                **Parameters**:
                - Destination: {destination}
                - Duration: {duration} days
                - Interests: {interests} (e.g., history, food, adventure, relaxation)
                
                {self.__tip_section()}
                
                Ensure the itinerary is realistic regarding travel times and activity durations. Balance scheduled activities with free time.
                """
            ),
            expected_output="A complete day-by-day itinerary with all requested details and practical considerations",
            agent=agent,
        )

    def provide_local_insights(self, agent, destination):
        return Task(
            description=dedent(
                f"""
                **Task**: Provide insider knowledge and local recommendations
                **Description**: As an experienced local guide, provide authentic insights and recommendations for travelers 
                visiting {destination}. Go beyond typical tourist information to share hidden gems, local customs, and practical 
                advice that will enhance the traveler's experience.
                
                Your insights should include:
                1. Hidden gems and off-the-beaten-path attractions locals love
                2. Cultural norms, etiquette, and local customs visitors should be aware of
                3. Common tourist mistakes or scams to avoid
                4. Authentic local dining recommendations at various price points
                5. Practical tips for navigating public transportation
                6. Useful local phrases or language tips
                7. Best times of day to visit popular attractions to avoid crowds
                8. Local festivals or events happening during the travel period
                9. Safety tips specific to different neighborhoods
                
                **Parameters**:
                - Destination: {destination}
                
                {self.__tip_section()}
                
                Focus on providing authentic, current information that can't easily be found in standard guidebooks or websites.
                """
            ),
            expected_output="A comprehensive guide with insider knowledge, cultural insights, and practical local tips",
            agent=agent,
        )

