import re

def write_report_guardrail(output):
    try:
        output_str = output if isinstance(output, str) else output.raw
    except Exception as e:
        raise ValueError(f"Error retrieving the `raw` argument: {str(e)}")

    output_lower = output_str.lower()

    if not re.search(r'#+.*summary', output_lower):
        raise ValueError("The report must include a Summary section with a header like '## Summary'")

    if not re.search(r'#+.*insights|#+.*recommendations', output_lower):
        raise ValueError("The report must include an Insights section with a header like '## Insights' or '## Recommendations'")

    if not re.search(r'#+.*citations|#+.*references|#+.*sources', output_lower):
        raise ValueError("The report must include a Citations, References, or Sources section with a header like '## Citations' or '## Sources'")

    return output_str