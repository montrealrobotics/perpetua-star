
PROMPT_WEEKLY_SCHEDULE = """
Consider a household environment composed of two parents and two children. The 
environment contains various objects located in different places, where objects
can appear when needed for an activity (e.g., plates for eating) or disappear when 
taken by a member (e.g., keys/bags when leaving). During weekdays, the parents 
typically leave for work in the morning and return in the evening (07:30 - 18:00), 
while the children attend school during the day (08:00 - 17:00). On weekends, 
the family tends to stay home more often, engaging in activities such as cooking, 
cleaning, and leisure time together.

Task: Generate a weekly schedule (Mon-Sun) for the specific object at the specific location.

Output Rules:
- Output ONLY a Python list of strings.
- Use 3-letter day abbreviations (Mon, Tue, Wed, Thu, Fri, Sat, Sun).
- Format: ['Day: HH:MM - Day: HH:MM', 'Day: HH:MM - Day: HH:MM', ...]
- Do not output markdown code blocks, explanations, or context.

Example 1:
Input:
    Object: Plates
    Location: On Dinner Table
    Type: Active Location (appear when used)
Output: 
    Schedule: ['Mon: 19:00 - Mon: 19:30', 'Tue: 19:00 - Tue: 19:30', 'Wed: 19:00 - Wed: 19:30', 'Thu: 19:00 - Thu: 19:30', 'Fri: 19:00 - Fri: 19:30', 'Sat: 12:30 - Sat: 13:00', 'Sat: 18:00 - Sat: 18:30', 'Sun: 12:30 - Sun: 13:00', 'Sun: 18:00 - Sun: 18:30']

Example 2:
Input:
    Object: RC Car
    Location: In Toy Box
    Type: Passive Location (disappear when used)
Output: 
    Schedule: ['Mon: 00:00 - Mon: 17:30', 'Mon: 20:00 - Tue: 17:30', 'Tue: 20:00 - Wed: 17:30', 'Wed: 20:00 - Thu: 17:30', 'Thu: 20:00 - Fri: 17:30', 'Fri: 21:00 - Sat: 10:00', 'Sat: 16:00 - Sun: 10:00', 'Sun: 16:00 - Sun: 23:59']

Current task:

Input:
    Object: {object_name}
    Location: {object_location}
    Type: {object_type}
Output: 
"""


def get_weekly_prompt(
    object_name: str,
    object_location: str,
    object_type: str,
) -> str:
    return PROMPT_WEEKLY_SCHEDULE.format(
        object_name=object_name,
        object_location=object_location,
        object_type=object_type,
    )
