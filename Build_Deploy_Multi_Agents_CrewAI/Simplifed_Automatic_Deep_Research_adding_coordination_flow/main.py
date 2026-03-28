#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from .crew import LowCostResearchCrew


def run():
    inputs = {
        "user_query": "AI ethics"
    }

    result = LowCostResearchCrew().crew().kickoff(inputs=inputs)

    print("\n=== FINAL RESULT ===\n")
    print(result)


if __name__ == "__main__":
    run()
