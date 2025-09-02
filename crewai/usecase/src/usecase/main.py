#!/usr/bin/env python
import sys
import warnings
from datetime import datetime

from usecase.crew import TripPlanner  # updated import for your use case

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the TripPlanner crew.
    """
    inputs = {
        'days': 6,
        'budget': 'medium',
        'interests': ['cultural', 'nature'],
        'current_year': str(datetime.now().year)
    }
    
    try:
        TripPlanner().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


# def train():
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {
#         "days": 6,
#         "budget": "medium",
#         "interests": ["cultural", "nature"],
#         "current_year": str(datetime.now().year)
#     }
#     try:
#         TripPlanner().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
#     except Exception as e:
#         raise Exception(f"An error occurred while training the crew: {e}")

# def replay():
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         TripPlanner().crew().replay(task_id=sys.argv[1])
#     except Exception as e:
#         raise Exception(f"An error occurred while replaying the crew: {e}")

# def test():
#     """
#     Test the crew execution and return the results.
#     """
#     inputs = {
#         "days": 6,
#         "budget": "medium",
#         "interests": ["cultural", "nature"],
#         "current_year": str(datetime.now().year)
#     }
#     try:
#         TripPlanner().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
#     except Exception as e:
#         raise Exception(f"An error occurred while testing the crew: {e}")
